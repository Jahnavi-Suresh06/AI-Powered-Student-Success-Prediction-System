# 🎓 AI-Powered Student Success Prediction & Analytics System

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-89.50%25-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> A machine learning system that predicts student academic performance and delivers
> personalised, prioritised improvement recommendations through an interactive dashboard.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Dashboard Pages](#dashboard-pages)
- [Interview Q&A](#interview-qa)

---

## Overview

Traditional academic assessment identifies struggling students **after** performance has
already declined. This system provides **early, proactive detection** by analysing key
academic signals and predicting one of four performance categories:

| Category | Description |
|---|---|
| 🌟 Excellent | Top-performing student, on track for distinction |
| 👍 Good | Above-average, minor improvements possible |
| 📈 Average | At the midpoint, targeted effort needed |
| 🆘 At Risk | Immediate intervention required |

Once predicted, the **recommendation engine** generates a personalised, prioritised
3-step action plan based on the student's specific weak areas.

---

## Features

- **ML Prediction** — Random Forest classifier with 89.50% accuracy
- **Confidence Score** — Probability breakdown across all 4 categories
- **Recommendation Engine** — Rule-based, prioritised, personalised tips
- **EDA Analytics** — 8 interactive visualisation charts
- **Streamlit Dashboard** — 4-page professional UI, no technical knowledge needed
- **Dataset Explorer** — Filterable table with live summary statistics

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.x | Core language |
| Pandas | 2.1.0 | Data manipulation |
| NumPy | 1.24.3 | Numerical operations |
| Scikit-learn | 1.3.0 | ML model, evaluation, GridSearchCV |
| Matplotlib | 3.7.2 | Static EDA charts |
| Seaborn | 0.12.2 | Statistical visualisations |
| Plotly | 5.15.0 | Interactive dashboard charts |
| Streamlit | 1.25.0 | Web dashboard framework |
| Joblib | 1.3.1 | Model persistence |

---

## Project Structure

```
student_success_predictor/
│
├── data/
│   ├── raw/                        # Original synthetic dataset (1000 records)
│   └── processed/                  # Cleaned data + all EDA charts (.png, .html)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Dataset generation, cleaning, feature engineering
│   ├── model_training.py           # Model comparison, tuning, evaluation, saving
│   ├── recommendation.py           # Rule-based personalised recommendation engine
│   └── visualizations.py           # All 8 EDA chart generation functions
│
├── models/
│   ├── random_forest_model.pkl     # Trained Random Forest model
│   └── feature_columns.pkl         # Saved feature column order
│
├── app/
│   └── dashboard.py                # Main Streamlit 4-page dashboard
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.x installed
- Windows 10/11 with CMD or PowerShell

### Step 1 — Clone or download the project

```cmd
cd C:\Users\YourName
git clone https://github.com/yourusername/student_success_predictor.git
cd student_success_predictor
```

### Step 2 — Create and activate virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```cmd
pip install -r requirements.txt
```

---

## How to Run

Run each script **in order** from the project root directory:

```cmd
:: Step 1 — Generate and clean the dataset
python src\data_preprocessing.py

:: Step 2 — Generate all EDA visualisations
python src\visualizations.py

:: Step 3 — Train and save the ML model
python src\model_training.py

:: Step 4 — Test the recommendation engine
python src\recommendation.py

:: Step 5 — Launch the Streamlit dashboard
streamlit run app\dashboard.py
```

Then open your browser at: **http://localhost:8501**

---

## Model Performance

### Algorithm Comparison

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 76.37% | 73.50% |
| Decision Tree | 81.50% | 83.50% |
| **Random Forest** | **86.88%** | **90.00%** |
| Gradient Boosting | 86.63% | 89.00% |

### Final Model — Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| At Risk | 0.91 | **0.95** | 0.93 |
| Average | 0.88 | 0.88 | 0.88 |
| Good | 0.87 | 0.87 | 0.87 |
| Excellent | 0.93 | 0.92 | 0.92 |
| **Overall** | **0.90** | **0.90** | **0.89** |

> At Risk recall of **0.95** is the most critical metric — the system correctly
> identifies 95% of genuinely struggling students before it is too late.

### Optimal Hyperparameters (GridSearchCV)

```
n_estimators      : 200
max_depth         : 20
min_samples_split : 2
min_samples_leaf  : 1
```

---

## Dashboard Pages

### 🏠 Home
- Live KPI cards: total students, excellent count, at-risk count, avg CGPA
- Interactive donut chart of performance distribution
- Average CGPA by category bar chart
- "How it works" 4-step explainer

### 🔮 Predict Performance
- Input form with sliders for all academic metrics
- Instant prediction with confidence score
- Probability breakdown chart across all 4 categories
- Personalised recommendations with expandable action steps
- Benchmark gap chart vs Excellent student profile

### 📊 Analytics Dashboard
- **Tab 1:** Attendance and study hours histograms; CGPA and marks boxplots
- **Tab 2:** Full correlation heatmap; feature-to-target correlation bar chart
- **Tab 3:** Study hours vs CGPA scatter; grouped metrics bar chart

### 📋 Dataset Explorer
- Multi-filter table (by category, CGPA range, attendance range)
- Live row count as filters applied
- Summary statistics panel

---

## Interview Q&A

**Q: Why Random Forest over other models?**
> Random Forest consistently achieved the highest test accuracy (90%) and lowest
> cross-validation variance. It handles non-linear relationships, is robust to outliers,
> requires minimal preprocessing, and provides built-in feature importance rankings.

**Q: What does 89.5% accuracy mean in practice?**
> For a 4-class classification problem, this is strong performance. More importantly,
> the At Risk category achieves 0.95 recall — the system correctly flags 95% of
> genuinely struggling students, which is the metric that matters most in an
> early-warning academic system.

**Q: How does the recommendation engine work?**
> It is a rule-based expert system. Each academic metric is evaluated against three
> threshold levels (critical, low, good). Weaknesses are scored by severity and the
> top 3 are surfaced as a prioritised action plan. This design was chosen over an ML
> approach because recommendations need to be transparent and explainable to students
> and advisors.

**Q: How would you deploy this in production?**
> Push the repository to GitHub, connect it to Streamlit Community Cloud for free
> public deployment. For a production system with authentication and a real database,
> containerise with Docker and deploy on AWS EC2 or Azure App Service.

---

## Future Scope

- Integration with real college ERP/LMS for live data feeds
- LSTM-based temporal modelling across multiple semesters
- Feedback loop with reinforcement learning to optimise recommendations
- Mobile app with push notifications for at-risk alerts
- Role-based access: student, faculty, and HOD dashboards

---

*Final Year Project — BE Artificial Intelligence & Data Science | 2024–2025*
To run the code :
 streamlit run app\dashboard.py