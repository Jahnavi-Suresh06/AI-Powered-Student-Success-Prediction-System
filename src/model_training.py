# =============================================================
# FILE: src/model_training.py
# PURPOSE: Train, evaluate, and save the ML model
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib    # For saving and loading trained models
import os
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD AND PREPARE DATA
# ─────────────────────────────────────────────


def load_and_prepare_data():
    """
    Loads the cleaned dataset and prepares features
    and target variable for model training.

    Key concept — Features vs Target:
    - Features (X): The inputs the model learns from
    - Target  (y): What the model tries to predict
    """
    data_path = os.path.join('data', 'processed', 'student_cleaned.csv')
    df = pd.read_csv(data_path)

    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # ── Select Feature Columns (X) ───────────────────────────
    # We exclude: student_id (not useful), performance_category
    # (text version of target), performance_encoded (the target itself)
    feature_cols = [
        'attendance_percentage',
        'study_hours_per_day',
        'assignment_completion_rate',
        'internal_marks',
        'previous_cgpa',
        'extracurricular_activities',
        'part_time_job',
        'academic_score_index',
        'engagement_score',
        'study_efficiency'
    ]

    # Handle gender columns — they may or may not exist after encoding
    gender_cols = [c for c in df.columns if c.startswith('gender_')]
    feature_cols += gender_cols

    # Only use columns that actually exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df['performance_encoded'].copy()

    # ── Handle any remaining NaN values ──────────────────────
    # Safety net: fill with median before passing to model
    X = X.fillna(X.median())

    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    print(f"  (0=At Risk, 1=Average, 2=Good, 3=Excellent)\n")

    return X, y, feature_cols


# ─────────────────────────────────────────────
# STEP 2: SPLIT DATA INTO TRAIN AND TEST SETS
# ─────────────────────────────────────────────

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training set (80%) and test set (20%).

    WHY SPLIT?
    We train the model on 80% of data, then test it on
    the remaining 20% that the model has NEVER seen.
    This gives an honest measure of real-world performance.

    If we tested on training data, the model would appear
    100% accurate — but fail on new students. This is
    called OVERFITTING.

    test_size=0.2  → 20% of data is kept for testing
    random_state=42 → ensures same split every run
    stratify=y     → ensures all 4 categories are
                     proportionally represented in
                     both train and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y        # Critical: maintains class proportions
    )

    print(f"Training set size : {X_train.shape[0]} samples")
    print(f"Test set size     : {X_test.shape[0]} samples")
    print(
        f"Train/Test split  : {100-test_size*100:.0f}% / {test_size*100:.0f}%\n")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 3: COMPARE MULTIPLE MODELS
# ─────────────────────────────────────────────

def compare_models(X_train, X_test, y_train, y_test):
    """
    Trains 4 different models and compares their accuracy.

    Industry best practice: NEVER assume one model is best.
    Always compare multiple algorithms on the same data.
    This is called model selection.

    We use cross-validation (cv=5) which means:
    - Split training data into 5 equal parts
    - Train on 4 parts, test on 1 part
    - Repeat 5 times, take the average accuracy
    - Much more reliable than a single train/test split
    """

    print("=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )
    }

    results = {}

    print(f"\n{'Model':<25} {'CV Accuracy':>12} {'Test Accuracy':>14}")
    print("-" * 55)

    for name, model in models.items():
        # Cross-validation score on training data
        # cv=5 means 5-fold cross validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='accuracy'
        )

        # Train on full training set, evaluate on test set
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_acc': test_acc
        }

        print(
            f"{name:<25} {cv_scores.mean():.4f} ± {cv_scores.std():.3f}   {test_acc:.4f}")

    # Plot comparison chart
    _plot_model_comparison(results)

    # Return the best model based on test accuracy
    best_name = max(results, key=lambda k: results[k]['test_acc'])
    print(f"\nBest model: {best_name} "
          f"(Test Accuracy: {results[best_name]['test_acc']:.4f})")

    return results, best_name


def _plot_model_comparison(results):
    """Bar chart comparing all model accuracies."""
    names = list(results.keys())
    test_accs = [results[n]['test_acc'] for n in names]
    cv_means = [results[n]['cv_mean'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Model Accuracy Comparison', fontsize=16, fontweight='bold')

    bars1 = ax.bar(x - width/2, cv_means,  width, label='CV Accuracy (Train)',
                   color='#3498db', edgecolor='white', linewidth=1.2)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy',
                   color='#2ecc71', edgecolor='white', linewidth=1.2)

    # Add value labels on bars
    for bar in bars1 + bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Accuracy Score')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.axhline(y=0.8, color='red', linestyle='--',
               alpha=0.5, linewidth=1, label='80% threshold')

    plt.tight_layout()
    out = os.path.join('data', 'processed', 'model_comparison.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=120)
    plt.show(block=False)
    plt.pause(4)
    plt.close(fig)
    print(f"    Chart saved: {out}")


# ─────────────────────────────────────────────
# STEP 4: TUNE THE BEST MODEL
# ─────────────────────────────────────────────

def tune_random_forest(X_train, y_train):
    """
    Hyperparameter Tuning using GridSearchCV.

    WHAT ARE HYPERPARAMETERS?
    These are settings you configure BEFORE training —
    the model cannot learn them from data itself.
    Example: how many trees to use in the forest.

    GridSearchCV tries every combination of parameters
    and picks the best one using cross-validation.

    Think of it like trying every combination on a lock
    to find which one opens it.
    """
    print("\n" + "=" * 55)
    print("  HYPERPARAMETER TUNING (Random Forest)")
    print("  This may take 1-2 minutes...")
    print("=" * 55)

    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [100, 200],       # Number of trees
        'max_depth': [10, 20, None],   # Max depth of each tree
        'min_samples_split': [2, 5],           # Min samples to split a node
        'min_samples_leaf': [1, 2]            # Min samples in leaf node
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # cv=3 to keep it fast; use cv=5 for production
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param:<25} : {value}")
    print(f"\nBest CV Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# ─────────────────────────────────────────────
# STEP 5: EVALUATE THE FINAL MODEL
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation.

    METRICS EXPLAINED (in simple terms):

    ACCURACY: Out of all predictions, how many were correct?
      → "The model was correct 89% of the time"

    PRECISION: When the model predicts 'Excellent',
               how often is it actually correct?
      → "Of students predicted Excellent, 91% truly were"

    RECALL: Of all truly 'Excellent' students,
            how many did the model find?
      → "The model found 88% of all actual Excellent students"

    F1-SCORE: Harmonic mean of Precision and Recall.
              The best single number to judge a model.
      → "Balanced score combining precision and recall"

    CONFUSION MATRIX: A table showing what the model
    predicted vs what the actual label was.
    The diagonal = correct predictions.
    Off-diagonal = mistakes.
    """
    label_names = ['At Risk', 'Average', 'Good', 'Excellent']
    y_pred = model.predict(X_test)

    print("\n" + "=" * 55)
    print("  FINAL MODEL EVALUATION")
    print("=" * 55)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy : {acc:.4f} ({acc*100:.2f}%)")

    print("\nDetailed Classification Report:")
    print("-" * 55)
    print(classification_report(
        y_test, y_pred,
        target_names=label_names
    ))

    # ── Confusion Matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Evaluation — Random Forest',
                 fontsize=16, fontweight='bold')

    # Left: Confusion Matrix (counts)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Right: Confusion Matrix (percentages)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Greens',
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1], linewidths=0.5, cbar=False
    )
    axes[1].set_title('Confusion Matrix (%)')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    out = os.path.join('data', 'processed', 'confusion_matrix.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=120)
    plt.show(block=False)
    plt.pause(4)
    plt.close(fig)
    print(f"\n    Chart saved: {out}")

    return acc, y_pred


# ─────────────────────────────────────────────
# STEP 6: FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_cols):
    """
    Random Forest tells us which features it relied
    on most when making predictions. This is called
    Feature Importance.

    GREAT INTERVIEW TALKING POINT:
    "I used feature importance from Random Forest to
    validate my EDA findings. Previous CGPA and internal
    marks ranked highest, confirming they are the
    strongest predictors of student performance."
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Clean feature names for display
    clean_names = [f.replace('_', ' ').title() for f in feature_cols]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Feature Importance — Random Forest',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.RdYlGn(
        np.linspace(0.2, 0.9, len(feature_cols))
    )[::-1]

    bars = ax.barh(
        [clean_names[i] for i in indices],
        importances[indices],
        color=colors,
        edgecolor='white',
        linewidth=0.8
    )

    # Add value labels
    for bar, val in zip(bars, importances[indices]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)

    ax.set_xlabel('Importance Score (higher = more predictive)')
    ax.set_xlim(0, max(importances) * 1.2)
    ax.invert_yaxis()   # Most important at the top

    plt.tight_layout()
    out = os.path.join('data', 'processed', 'feature_importance.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=120)
    plt.show(block=False)
    plt.pause(4)
    plt.close(fig)
    print(f"    Feature importance chart saved: {out}")


# ─────────────────────────────────────────────
# STEP 7: SAVE THE TRAINED MODEL
# ─────────────────────────────────────────────

def save_model(model, scaler=None):
    """
    Saves the trained model to disk using joblib.

    WHY SAVE THE MODEL?
    Training takes time. Once trained, we save the model
    so the Streamlit dashboard can load it instantly
    and make predictions without retraining every time.

    .pkl = "pickle" format — Python's way of serializing
    any object to a file.
    """
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')

    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")

    # Save feature column names so the dashboard
    # knows exactly what input order to use
    feature_path = os.path.join(model_dir, 'feature_columns.pkl')
    return model_path


def save_feature_cols(feature_cols):
    """Saves the feature column list alongside the model."""
    path = os.path.join('models', 'feature_columns.pkl')
    joblib.dump(feature_cols, path)
    print(f"Feature columns saved: {path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_training_pipeline():
    """
    Runs the complete ML training pipeline:
    Load → Split → Compare → Tune → Evaluate → Save
    """

    print("\n" + "=" * 55)
    print("  STUDENT SUCCESS PREDICTOR")
    print("  Machine Learning Training Pipeline")
    print("=" * 55 + "\n")

    # Step 1: Load data
    X, y, feature_cols = load_and_prepare_data()

    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Compare models
    results, best_model_name = compare_models(
        X_train, X_test, y_train, y_test
    )

    # Step 4: Tune Random Forest
    best_model = tune_random_forest(X_train, y_train)

    # Step 5: Evaluate
    accuracy, y_pred = evaluate_model(best_model, X_test, y_test)

    # Step 6: Feature importance
    plot_feature_importance(best_model, feature_cols)

    # Step 7: Save
    save_model(best_model)
    save_feature_cols(feature_cols)

    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE")
    print(f"  Final Model Accuracy : {accuracy*100:.2f}%")
    print(f"  Model saved to       : models\\random_forest_model.pkl")
    print("=" * 55)

    return best_model, feature_cols


if __name__ == "__main__":
    model, feature_cols = run_training_pipeline()
