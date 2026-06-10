# =============================================================
# FILE: src/data_preprocessing.py
# PURPOSE: Generate synthetic dataset + clean and preprocess it
# AUTHOR: Your Name
# =============================================================

import pandas as pd          # For working with tables (DataFrames)
import numpy as np            # For numerical operations and random data
import os                     # For file path and directory operations


# ─────────────────────────────────────────────
# SECTION 1: DATASET GENERATION
# ─────────────────────────────────────────────

def generate_student_dataset(n_students=1000, random_seed=42):
    """
    Generates a realistic synthetic student dataset.

    Parameters:
        n_students (int): Number of student records to generate
        random_seed (int): Seed for reproducibility

    Returns:
        pd.DataFrame: A DataFrame with all student features
    """

    # Setting a seed means every time you run this code,
    # you get the SAME random numbers. This is critical for
    # reproducibility — a key concept in data science.
    np.random.seed(random_seed)

    # ── Student IDs ──────────────────────────────────────────
    # Creates IDs like S0001, S0002, ..., S1000
    student_ids = [f"S{str(i).zfill(4)}" for i in range(1, n_students + 1)]

    # ── Age ──────────────────────────────────────────────────
    # randint(low, high, size) — generates random integers
    # Most engineering students are between 18 and 23
    age = np.random.randint(18, 24, n_students)

    # ── Gender ───────────────────────────────────────────────
    # np.random.choice picks randomly from a list
    # p= sets the probability for each option (must sum to 1.0)
    gender = np.random.choice(
        ['Male', 'Female', 'Other'],
        n_students,
        p=[0.52, 0.45, 0.03]
    )

    # ── Attendance Percentage ────────────────────────────────
    # np.random.normal(mean, std_deviation, size)
    # Most students have ~75% attendance, with natural variation
    # clip() ensures values stay within realistic bounds [30, 100]
    attendance = np.clip(
        np.random.normal(loc=75, scale=15, size=n_students),
        30, 100
    ).round(1)

    # ── Study Hours Per Day ──────────────────────────────────
    # Average student studies about 4 hours/day
    study_hours = np.clip(
        np.random.normal(loc=4, scale=2, size=n_students),
        0, 12
    ).round(1)

    # ── Assignment Completion Rate ───────────────────────────
    # Correlated slightly with study hours (harder workers complete more)
    # We add study_hours * 2 to create a realistic positive correlation
    assignment_rate = np.clip(
        np.random.normal(loc=75, scale=15, size=n_students) +
        (study_hours * 2),
        40, 100
    ).round(1)

    # ── Internal Marks ───────────────────────────────────────
    # Correlated with both attendance and study hours
    internal_marks = np.clip(
        (attendance * 0.3) +
        (study_hours * 3) +
        np.random.normal(loc=20, scale=8, size=n_students),
        20, 100
    ).round(1)

    # ── Previous CGPA ────────────────────────────────────────
    # Indian university CGPA scale: 0 to 10
    # Correlated with internal marks (good students stay good)
    previous_cgpa = np.clip(
        (internal_marks / 100 * 6) +
        np.random.normal(loc=2, scale=0.8, size=n_students),
        4.0, 10.0
    ).round(2)

    # ── Extracurricular Activities ───────────────────────────
    # Binary: 1 = participates, 0 = does not
    extracurricular = np.random.choice([0, 1], n_students, p=[0.4, 0.6])

    # ── Part-Time Job ────────────────────────────────────────
    # Binary: 1 = has a job, 0 = no job
    part_time_job = np.random.choice([0, 1], n_students, p=[0.7, 0.3])

    # ── Build the DataFrame ──────────────────────────────────
    # A DataFrame is like an Excel sheet in Python
    df = pd.DataFrame({
        'student_id': student_ids,
        'age': age,
        'gender': gender,
        'attendance_percentage': attendance,
        'study_hours_per_day': study_hours,
        'assignment_completion_rate': assignment_rate,
        'internal_marks': internal_marks,
        'previous_cgpa': previous_cgpa,
        'extracurricular_activities': extracurricular,
        'part_time_job': part_time_job
    })

    # ── Create Performance Category (Target Label) ───────────
    # This is the column our ML model will learn to predict.
    # We define rules based on combined academic signals.
    df['performance_category'] = df.apply(
        _assign_performance_category, axis=1
    )

    return df


def _assign_performance_category(row):
    """
    Assigns a performance label using percentile-based
    thresholds so categories are naturally balanced.
    """
    score = (
        (row['previous_cgpa'] / 10.0) * 35 +
        (row['internal_marks'] / 100.0) * 25 +
        (row['attendance_percentage'] / 100) * 20 +
        (row['study_hours_per_day'] / 12.0) * 10 +
        (row['assignment_completion_rate'] / 100) * 10
    )

    # Recalibrated thresholds based on actual score distribution
    # Score range in our data is roughly 42–72
    if score >= 63:
        return 'Excellent'
    elif score >= 56:
        return 'Good'
    elif score >= 49:
        return 'Average'
    else:
        return 'At Risk'


# ─────────────────────────────────────────────
# SECTION 2: DATA CLEANING
# ─────────────────────────────────────────────

def introduce_missing_values(df, missing_rate=0.03):
    """
    Introduces a small % of missing values to simulate
    real-world data quality issues.

    In real datasets, data entry errors and system failures
    cause missing values. We add them intentionally so we
    can practice handling them professionally.
    """
    df_missing = df.copy()  # Never modify the original DataFrame

    # Columns that realistically could have missing data
    cols_with_missing = [
        'attendance_percentage',
        'study_hours_per_day',
        'assignment_completion_rate',
        'internal_marks',
        'previous_cgpa'
    ]

    for col in cols_with_missing:
        # Randomly select row indices to set as NaN (missing)
        missing_indices = np.random.choice(
            df_missing.index,
            size=int(len(df_missing) * missing_rate),
            replace=False
        )
        df_missing.loc[missing_indices, col] = np.nan

    return df_missing


def clean_dataset(df):
    """
    Performs professional data cleaning:
    1. Reports missing values
    2. Fills missing values with appropriate strategies
    3. Removes duplicate records
    4. Validates data ranges
    5. Encodes categorical variables

    Returns:
        pd.DataFrame: Cleaned and encoded DataFrame
    """

    print("=" * 55)
    print("  DATA CLEANING REPORT")
    print("=" * 55)

    # ── Step 1: Check Missing Values ─────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    print("\n[1] Missing Values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"    {col:<35} {missing[col]} ({missing_pct[col]}%)")

    # ── Step 2: Fill Missing Values ──────────────────────────
    # Industry best practice:
    # - Numerical columns with normal distribution → fill with MEDIAN
    #   (Median is robust to outliers; Mean can be skewed)
    # - Categorical columns → fill with MODE (most common value)

    numerical_cols = [
        'attendance_percentage',
        'study_hours_per_day',
        'assignment_completion_rate',
        'internal_marks',
        'previous_cgpa'
    ]

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(
                f"\n    Filled '{col}' missing values with median: {median_val}")

    # ── Step 3: Remove Duplicates ────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"\n[2] Duplicates removed: {before - after}")

    # ── Step 4: Validate Data Ranges ─────────────────────────
    # Catches any values that slipped outside expected bounds
    print("\n[3] Range Validation:")

    range_rules = {
        'attendance_percentage': (0, 100),
        'study_hours_per_day': (0, 12),
        'assignment_completion_rate': (0, 100),
        'internal_marks': (0, 100),
        'previous_cgpa': (0, 10)
    }

    for col, (low, high) in range_rules.items():
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if len(out_of_range) > 0:
            print(
                f"    WARNING: {len(out_of_range)} out-of-range values in '{col}'")
            # Clip to valid range instead of dropping (preserves data)
            df[col] = df[col].clip(low, high)
        else:
            print(f"    OK: '{col}' values within [{low}, {high}]")

    # ── Step 5: Encode Categorical Variables ─────────────────
    # ML models only understand numbers, not text like "Male".
    # We convert text categories into numbers.

    print("\n[4] Encoding Categorical Variables:")

    # Gender: One-Hot Encoding
    # Creates new columns: gender_Female, gender_Male, gender_Other
    # drop_first=True removes one column to avoid multicollinearity
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    print("    Gender → One-Hot Encoded (gender_Female, gender_Other added)")

    # Performance Category: Label Encoding (for the target column)
    # We map text labels to numbers that have a meaningful order
    performance_map = {
        'At Risk': 0,
        'Average': 1,
        'Good': 2,
        'Excellent': 3
    }
    df['performance_encoded'] = df['performance_category'].map(performance_map)
    print("    Performance → Label Encoded (At Risk=0, Average=1, Good=2, Excellent=3)")

    print("\n" + "=" * 55)
    print(f"  CLEANED DATASET: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 55)

    return df


# ─────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    """
    Feature Engineering: Creating new meaningful columns
    from existing ones.

    Why? Sometimes combining features gives the model
    more signal than individual features alone.
    This is a key skill that separates good data scientists
    from great ones.
    """

    print("\n[5] Feature Engineering:")

    # Academic Score Index: a single combined metric
    # This helps the model understand overall academic health
    df['academic_score_index'] = (
        (df['attendance_percentage'] * 0.25) +
        (df['internal_marks'] * 0.35) +
        (df['assignment_completion_rate'] * 0.20) +
        (df['study_hours_per_day'] / 12 * 100 * 0.20)
    ).round(2)

    print("    Created: 'academic_score_index'")

    # Engagement Score: measures how actively involved a student is
    df['engagement_score'] = (
        (df['assignment_completion_rate'] * 0.5) +
        (df['attendance_percentage'] * 0.5)
    ).round(2)

    print("    Created: 'engagement_score'")

    # Study Efficiency: CGPA earned per hour of study
    # Avoids division by zero with a small epsilon (0.1)
    df['study_efficiency'] = (
        df['previous_cgpa'] / (df['study_hours_per_day'] + 0.1)
    ).round(3)

    print("    Created: 'study_efficiency'")

    return df


# ─────────────────────────────────────────────
# SECTION 4: MAIN PIPELINE — RUN EVERYTHING
# ─────────────────────────────────────────────

def run_preprocessing_pipeline():
    """
    Master function that runs all steps in order:
    Generate → Introduce Missing → Clean → Engineer → Save
    """

    print("\n" + "=" * 55)
    print("  STUDENT SUCCESS PREDICTOR")
    print("  Data Preprocessing Pipeline")
    print("=" * 55 + "\n")

    # Step 1: Generate raw data
    print("Generating dataset...")
    df_raw = generate_student_dataset(n_students=1000)
    print(f"Raw dataset shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}\n")

    # Step 2: Save raw data (always save raw data untouched)
    raw_path = os.path.join('data', 'raw', 'student_data.csv')
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw data saved to: {raw_path}")

    # Step 3: Introduce realistic missing values
    df_with_missing = introduce_missing_values(df_raw, missing_rate=0.03)

    # Step 4: Clean the dataset
    df_cleaned = clean_dataset(df_with_missing)

    # Step 5: Engineer new features
    df_final = engineer_features(df_cleaned)

    # Step 6: Save processed data
    processed_path = os.path.join('data', 'processed', 'student_cleaned.csv')
    df_final.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to: {processed_path}")

    # Step 7: Final summary
    print("\n" + "=" * 55)
    print("  PREPROCESSING COMPLETE")
    print(
        f"  Final shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print("=" * 55)

    # Show class distribution of target variable
    print("\nTarget Variable Distribution:")
    dist = df_final['performance_category'].value_counts()
    for category, count in dist.items():
        pct = count / len(df_final) * 100
        bar = "█" * int(pct / 2)
        print(f"  {category:<12} {bar:<25} {count} ({pct:.1f}%)")

    return df_final


# ─────────────────────────────────────────────
# This block runs ONLY when you execute this
# file directly (not when imported elsewhere)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    print("\nFirst 5 rows of final dataset:")
    print(df.head())
