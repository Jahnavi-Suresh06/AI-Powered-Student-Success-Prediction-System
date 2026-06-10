# =============================================================
# FILE: src/visualizations.py
# PURPOSE: All EDA charts and visualizations for the project
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings('ignore')  # Suppress minor warnings


# ─────────────────────────────────────────────
# GLOBAL STYLE SETTINGS
# Industry practice: define your visual style
# once at the top, apply everywhere.
# ─────────────────────────────────────────────

# Color palette for our 4 performance categories
# Using a traffic-light logic: green=good, red=bad
CATEGORY_COLORS = {
    'Excellent': '#2ecc71',   # Green
    'Good': '#3498db',   # Blue
    'Average': '#f39c12',   # Orange
    'At Risk': '#e74c3c'    # Red
}

# Seaborn style for all matplotlib charts
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'


# ─────────────────────────────────────────────
# CHART 1: TARGET VARIABLE DISTRIBUTION
# ─────────────────────────────────────────────

def plot_target_distribution(df):
    """
    Shows how many students fall into each
    performance category. This is the FIRST chart
    any data scientist checks — is the dataset balanced?

    INSIGHT TO LOOK FOR:
    If one category has 90% of records, the model will
    be biased toward predicting that category. That is
    called class imbalance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Student Performance Category Distribution',
                 fontsize=16, fontweight='bold')

    counts = df['performance_category'].value_counts()
    categories = counts.index.tolist()
    colors = [CATEGORY_COLORS[c] for c in categories]

    # ── Left: Bar Chart ──────────────────────────────────────
    bars = axes[0].bar(categories, counts.values, color=colors,
                       edgecolor='white', linewidth=1.5, width=0.6)
    axes[0].set_title('Count per Category')
    axes[0].set_xlabel('Performance Category')
    axes[0].set_ylabel('Number of Students')

    # Add value labels on top of each bar
    for bar, count in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            str(count),
            ha='center', va='bottom', fontweight='bold', fontsize=12
        )

    # ── Right: Pie Chart ─────────────────────────────────────
    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels=categories,
        colors=colors,
        autopct='%1.1f%%',     # Show percentage inside slices
        startangle=90,          # Start from top
        pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for text in autotexts:
        text.set_fontsize(11)
        text.set_fontweight('bold')

    axes[1].set_title('Percentage Distribution')

    plt.tight_layout()
    _save_and_show(fig, 'target_distribution.png')


# ─────────────────────────────────────────────
# CHART 2: FEATURE DISTRIBUTIONS (HISTOGRAMS)
# ─────────────────────────────────────────────

def plot_feature_distributions(df):
    """
    Plots histograms for all numerical features.
    This tells us:
    - Is the data normally distributed?
    - Are there outliers?
    - Is the data skewed left or right?

    INSIGHT: A normal distribution (bell curve) means
    most students cluster around the average. Heavy
    skew may require log transformation.
    """
    numerical_features = [
        'attendance_percentage',
        'study_hours_per_day',
        'assignment_completion_rate',
        'internal_marks',
        'previous_cgpa',
        'academic_score_index'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution of All Numerical Features',
                 fontsize=16, fontweight='bold')

    # Flatten the 2D array of axes into 1D for easy iteration
    axes = axes.flatten()

    for i, feature in enumerate(numerical_features):
        ax = axes[i]

        # KDE = Kernel Density Estimate: smooth curve over the histogram
        ax.hist(df[feature], bins=30, color='#3498db',
                alpha=0.7, edgecolor='white', linewidth=0.5)

        # Add vertical lines for mean and median
        mean_val = df[feature].mean()
        median_val = df[feature].median()

        ax.axvline(mean_val,   color='#e74c3c', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='#2ecc71', linestyle='-.',
                   linewidth=2, label=f'Median: {median_val:.1f}')

        ax.set_title(feature.replace('_', ' ').title())
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save_and_show(fig, 'feature_distributions.png')


# ─────────────────────────────────────────────
# CHART 3: BOXPLOTS BY PERFORMANCE CATEGORY
# ─────────────────────────────────────────────

def plot_boxplots_by_category(df):
    """
    Boxplots compare feature distributions ACROSS
    performance categories.

    HOW TO READ A BOXPLOT:
    - The box covers the middle 50% of data (IQR)
    - The line inside the box is the median
    - The whiskers extend to min/max (excluding outliers)
    - Dots outside whiskers are outliers

    INSIGHT: If 'Excellent' students have noticeably
    higher study_hours than 'At Risk' students, that
    feature is a strong predictor.
    """
    features_to_plot = [
        'attendance_percentage',
        'study_hours_per_day',
        'internal_marks',
        'previous_cgpa'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'Feature Distribution by Performance Category',
        fontsize=16, fontweight='bold'
    )
    axes = axes.flatten()

    category_order = ['At Risk', 'Average', 'Good', 'Excellent']
    palette = [CATEGORY_COLORS[c] for c in category_order]

    for i, feature in enumerate(features_to_plot):
        ax = axes[i]

        # seaborn boxplot: x=category, y=feature value
        sns.boxplot(
            data=df,
            x='performance_category',
            y=feature,
            order=category_order,
            palette=palette,
            ax=ax,
            linewidth=1.5,
            flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5}
        )

        ax.set_title(feature.replace('_', ' ').title())
        ax.set_xlabel('Performance Category')
        ax.set_ylabel(feature.replace('_', ' ').title())

        # Add mean dots on top of each box
        means = df.groupby('performance_category')[feature].mean()
        for j, cat in enumerate(category_order):
            ax.plot(j, means[cat], marker='D', color='white',
                    markersize=6, zorder=5, markeredgecolor='black')

    plt.tight_layout()
    _save_and_show(fig, 'boxplots_by_category.png')


# ─────────────────────────────────────────────
# CHART 4: CORRELATION HEATMAP
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df):
    """
    A correlation heatmap shows the relationship
    between EVERY pair of numerical features.

    CORRELATION VALUES:
    +1.0 = perfectly positively correlated
     0.0 = no relationship
    -1.0 = perfectly negatively correlated

    WHAT TO LOOK FOR:
    - Features strongly correlated with 'performance_encoded'
      are the best predictors for our model.
    - Features correlated with EACH OTHER may cause
      multicollinearity (a problem in some ML models).
    """
    # Select only numerical columns for correlation
    numerical_cols = [
        'attendance_percentage',
        'study_hours_per_day',
        'assignment_completion_rate',
        'internal_marks',
        'previous_cgpa',
        'academic_score_index',
        'engagement_score',
        'study_efficiency',
        'performance_encoded'
    ]

    # .corr() calculates Pearson correlation between all pairs
    corr_matrix = df[numerical_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(
        'Feature Correlation Heatmap',
        fontsize=16, fontweight='bold', y=1.01
    )

    # Create a mask for the upper triangle
    # (the heatmap is symmetric, no need to show both sides)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Draw the heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,           # Show correlation numbers in cells
        fmt='.2f',            # Round to 2 decimal places
        cmap='RdYlGn',        # Red=negative, Yellow=neutral, Green=positive
        center=0,             # Center colormap at 0
        vmin=-1, vmax=1,
        ax=ax,
        linewidths=0.5,
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'}
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    _save_and_show(fig, 'correlation_heatmap.png')


# ─────────────────────────────────────────────
# CHART 5: SCATTER PLOT — STUDY HOURS vs CGPA
# ─────────────────────────────────────────────

def plot_scatter_study_vs_cgpa(df):
    """
    Scatter plots reveal the relationship between
    two continuous variables. Color-coding by category
    adds a third dimension of insight.
    """

    # Fix: Drop rows where ANY of the columns we need have NaN
    # This prevents the 'size' property NaN error in plotly
    cols_needed = [
        'study_hours_per_day',
        'previous_cgpa',
        'performance_category',
        'internal_marks',
        'attendance_percentage'
    ]
    df_plot = df[cols_needed].dropna().copy()

    print(
        f"    Scatter plot using {len(df_plot)} clean rows (dropped NaN rows)")

    fig = px.scatter(
        df_plot,
        x='study_hours_per_day',
        y='previous_cgpa',
        color='performance_category',
        color_discrete_map=CATEGORY_COLORS,
        title='Study Hours per Day vs Previous CGPA',
        labels={
            'study_hours_per_day': 'Study Hours per Day',
            'previous_cgpa': 'Previous CGPA (out of 10)',
            'performance_category': 'Performance'
        },
        opacity=0.7,
        hover_data=['internal_marks', 'attendance_percentage'],
        size='internal_marks',
        size_max=15
    )

    fig.update_layout(
        font_family='Arial',
        title_font_size=16,
        legend_title_text='Performance Category',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
    )

    fig.write_html('data\\processed\\scatter_study_cgpa.html')
    fig.show()
    print("    Interactive chart saved: data\\processed\\scatter_study_cgpa.html")


# ─────────────────────────────────────────────
# CHART 6: ATTENDANCE vs PERFORMANCE (BAR)
# ─────────────────────────────────────────────

def plot_attendance_vs_performance(df):
    """
    Shows the average attendance for each performance
    category. The clearer the step-up from 'At Risk'
    to 'Excellent', the more predictive attendance is.
    """
    avg_attendance = df.groupby('performance_category')[
        'attendance_percentage'].mean()
    category_order = ['At Risk', 'Average', 'Good', 'Excellent']
    avg_attendance = avg_attendance.reindex(category_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Average Attendance by Performance Category',
                 fontsize=16, fontweight='bold')

    colors = [CATEGORY_COLORS[c] for c in category_order]
    bars = ax.bar(category_order, avg_attendance.values,
                  color=colors, edgecolor='white',
                  linewidth=1.5, width=0.5)

    # Add value labels inside bars
    for bar, val in zip(bars, avg_attendance.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f'{val:.1f}%',
            ha='center', va='center',
            color='white', fontweight='bold', fontsize=13
        )

    ax.set_xlabel('Performance Category', fontsize=12)
    ax.set_ylabel('Average Attendance (%)', fontsize=12)
    ax.set_ylim(0, 110)

    # Add a horizontal reference line at 75% (typical minimum requirement)
    ax.axhline(y=75, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.7, label='75% threshold')
    ax.legend()

    plt.tight_layout()
    _save_and_show(fig, 'attendance_vs_performance.png')


# ─────────────────────────────────────────────
# CHART 7: INTERACTIVE RADAR CHART
# ─────────────────────────────────────────────

def plot_radar_chart(df):
    """
    A radar (spider) chart shows the average profile
    of each performance category across multiple features.

    This is visually impressive and great for presentations.
    Each performance group gets its own polygon — the
    bigger and fuller the polygon, the stronger the student.
    """
    category_order = ['Excellent', 'Good', 'Average', 'At Risk']

    features = [
        'attendance_percentage',
        'study_hours_per_day',
        'assignment_completion_rate',
        'internal_marks',
        'previous_cgpa'
    ]

    # Normalize each feature to 0-100 scale for fair comparison
    df_norm = df.copy()
    df_norm['study_hours_per_day'] = df_norm['study_hours_per_day'] / 12 * 100
    df_norm['previous_cgpa'] = df_norm['previous_cgpa'] / 10 * 100

    # Calculate mean profile for each category
    category_profiles = df_norm.groupby(
        'performance_category')[features].mean()

    feature_labels = [
        'Attendance', 'Study Hours',
        'Assignment Rate', 'Internal Marks', 'CGPA'
    ]

    fig = go.Figure()

    for category in category_order:
        values = category_profiles.loc[category].tolist()
        values += values[:1]   # Close the radar polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_labels + [feature_labels[0]],
            fill='toself',
            fillcolor=CATEGORY_COLORS[category],
            opacity=0.2,
            line=dict(color=CATEGORY_COLORS[category], width=2.5),
            name=category
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        title=dict(
            text='Average Student Profile by Performance Category',
            font=dict(size=16)
        ),
        font=dict(family='Arial')
    )

    fig.write_html('data\\processed\\radar_chart.html')
    fig.show()
    print("    Interactive chart saved: data\\processed\\radar_chart.html")


# ─────────────────────────────────────────────
# CHART 8: FEATURE IMPORTANCE PREVIEW
# ─────────────────────────────────────────────

def plot_feature_correlation_with_target(df):
    """
    Shows how strongly each feature correlates
    with the target (performance_encoded).

    This is a preview of feature importance —
    features with high correlation are the ones
    our ML model will rely on most.

    GREAT INTERVIEW TALKING POINT:
    "Before modelling, I identified the top predictive
    features using correlation analysis, which informed
    my feature selection decisions."
    """
    numerical_cols = [
        'attendance_percentage',
        'study_hours_per_day',
        'assignment_completion_rate',
        'internal_marks',
        'previous_cgpa',
        'academic_score_index',
        'engagement_score',
        'study_efficiency'
    ]

    # Correlation of each feature with the target variable
    correlations = df[numerical_cols].corrwith(
        df['performance_encoded']
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        'Feature Correlation with Performance Score',
        fontsize=16, fontweight='bold'
    )

    # Color bars: green for positive correlation, red for negative
    bar_colors = ['#2ecc71' if v >
                  0 else '#e74c3c' for v in correlations.values]

    bars = ax.barh(
        [c.replace('_', ' ').title() for c in correlations.index],
        correlations.values,
        color=bar_colors,
        edgecolor='white',
        linewidth=1
    )

    # Add value labels on bars
    for bar, val in zip(bars, correlations.values):
        ax.text(
            val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}',
            va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=10, fontweight='bold'
        )

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    ax.set_xlim(-0.3, 1.1)

    # Add legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Positive correlation')
    red_patch = mpatches.Patch(color='#e74c3c', label='Negative correlation')
    ax.legend(handles=[green_patch, red_patch], loc='lower right')

    plt.tight_layout()
    _save_and_show(fig, 'feature_correlation_target.png')


# ─────────────────────────────────────────────
# HELPER FUNCTION: Save + Show Chart
# ─────────────────────────────────────────────

def _save_and_show(fig, filename):
    """
    Saves the chart as PNG and displays it.
    Increase pause seconds if you want more viewing time.
    """
    output_dir = os.path.join('data', 'processed')
    output_path = os.path.join(output_dir, filename)

    fig.savefig(output_path, bbox_inches='tight',
                facecolor='white', dpi=120)
    print(f"    Chart saved: {output_path}")

    plt.show(block=False)
    plt.pause(5)       # ← Change this number to however many
    #   seconds you want each chart visible
    plt.close(fig)         # Then close and move on


# ─────────────────────────────────────────────
# MAIN: RUN ALL EDA CHARTS
# ─────────────────────────────────────────────

def run_eda(df=None):
    """
    Runs the complete EDA pipeline.
    Loads data if not passed directly.
    """
    if df is None:
        data_path = os.path.join('data', 'processed', 'student_cleaned.csv')
        df = pd.read_csv(data_path)
        print(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    print("\n[1] Summary Statistics:")
    print(df.describe().round(2).to_string())

    print("\n\nGenerating Charts...")

    print("\n  Chart 1: Target Distribution")
    plot_target_distribution(df)

    print("\n  Chart 2: Feature Distributions")
    plot_feature_distributions(df)

    print("\n  Chart 3: Boxplots by Category")
    plot_boxplots_by_category(df)

    print("\n  Chart 4: Correlation Heatmap")
    plot_correlation_heatmap(df)

    print("\n  Chart 5: Study Hours vs CGPA Scatter")
    plot_scatter_study_vs_cgpa(df)

    print("\n  Chart 6: Attendance vs Performance")
    plot_attendance_vs_performance(df)

    print("\n  Chart 7: Radar Chart")
    plot_radar_chart(df)

    print("\n  Chart 8: Feature Correlation with Target")
    plot_feature_correlation_with_target(df)

    print("\n" + "=" * 55)
    print("  EDA COMPLETE — All charts saved to data\\processed\\")
    print("=" * 55)


if __name__ == "__main__":
    run_eda()
