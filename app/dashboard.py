# =============================================================
# FILE: app/dashboard.py
# PURPOSE: Main Streamlit dashboard for the project
# RUN WITH: streamlit run app/dashboard.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATH SETUP
# Adds the project root to Python's search path
# so we can import from the src/ folder
# ─────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommendation import generate_recommendations

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the FIRST Streamlit command called
# ─────────────────────────────────────────────
st.set_page_config(
    page_title = "Student Success Predictor",
    page_icon  = "🎓",
    layout     = "wide",          # Use full browser width
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# Injects CSS directly into Streamlit to style
# the dashboard beyond default Streamlit themes
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background and font */
    .main { background-color: #f8f9fa; }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #3498db;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-top: 4px;
    }

    /* Prediction result cards */
    .result-excellent {
        background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
        border-left: 6px solid #2ecc71;
        padding: 24px; border-radius: 12px;
        margin: 16px 0;
    }
    .result-good {
        background: linear-gradient(135deg, #d6eaf8, #a9cce3);
        border-left: 6px solid #3498db;
        padding: 24px; border-radius: 12px;
        margin: 16px 0;
    }
    .result-average {
        background: linear-gradient(135deg, #fef9e7, #fdebd0);
        border-left: 6px solid #f39c12;
        padding: 24px; border-radius: 12px;
        margin: 16px 0;
    }
    .result-at-risk {
        background: linear-gradient(135deg, #fdedec, #fadbd8);
        border-left: 6px solid #e74c3c;
        padding: 24px; border-radius: 12px;
        margin: 16px 0;
    }

    /* Recommendation cards */
    .rec-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
    }
    .rec-card-critical {
        border-left: 4px solid #e74c3c;
    }
    .rec-card-low {
        border-left: 4px solid #f39c12;
    }
    .rec-card-good {
        border-left: 4px solid #2ecc71;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        padding: 10px 0;
    }

    /* Priority badge */
    .priority-badge {
        display: inline-block;
        background: #e74c3c;
        color: white;
        border-radius: 50%;
        width: 28px; height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-right: 8px;
    }

    /* Hide Streamlit default menu */
    #MainMenu {visibility: hidden;}
    footer   {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA AND MODEL LOADING
# @st.cache_resource → loads once, stays in
# memory for all user interactions. Without this,
# the model would reload on every button click.
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Loads the trained Random Forest model."""
    model_path = os.path.join('models', 'random_forest_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_feature_cols():
    """Loads the saved feature column names."""
    path = os.path.join('models', 'feature_columns.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_dataset():
    """Loads the processed student dataset."""
    path = os.path.join('data', 'processed', 'student_cleaned.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">🎓 Student Success<br>Predictor</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        page = st.radio(
            "Navigate to:",
            options=[
                "🏠 Home",
                "🔮 Predict Performance",
                "📊 Analytics Dashboard",
                "📋 Dataset Explorer"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("**About this App**")
        st.caption(
            "An ML-powered system that predicts student "
            "academic performance and generates personalised "
            "improvement recommendations."
        )
        st.markdown("---")
        st.caption("Built with Python, Scikit-learn & Streamlit")
        st.caption("Phase 6 — Final Year Project")

    return page


# ─────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────

def render_home(df):
    # Hero section
    st.markdown("""
    <div style='text-align:center; padding: 30px 0 10px 0;'>
        <h1 style='font-size:2.8rem; color:#2c3e50; margin-bottom:8px;'>
            🎓 Student Success Prediction System
        </h1>
        <p style='font-size:1.2rem; color:#7f8c8d; max-width:700px;
                  margin:0 auto;'>
            An AI-powered analytics platform that predicts student academic
            performance and delivers personalised improvement recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Stats Row ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    if df is not None:
        total         = len(df)
        excellent_pct = len(df[df['performance_category'] == 'Excellent'])
        at_risk_count = len(df[df['performance_category'] == 'At Risk'])
        avg_cgpa      = df['previous_cgpa'].mean()
    else:
        total = excellent_pct = at_risk_count = 0
        avg_cgpa = 0

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{total}</p>
            <p class="metric-label">Total Students</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#2ecc71">
            <p class="metric-value">{excellent_pct}</p>
            <p class="metric-label">Excellent Performers</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#e74c3c">
            <p class="metric-value">{at_risk_count}</p>
            <p class="metric-label">At-Risk Students</p>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#9b59b6">
            <p class="metric-value">{avg_cgpa:.2f}</p>
            <p class="metric-label">Average CGPA</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two column layout for charts ─────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            '<div class="section-header">Performance Distribution</div>',
            unsafe_allow_html=True
        )
        if df is not None:
            counts = df['performance_category'].value_counts()
            colors = {
                'Excellent': '#2ecc71',
                'Good'     : '#3498db',
                'Average'  : '#f39c12',
                'At Risk'  : '#e74c3c'
            }
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map=colors,
                hole=0.45          # Makes it a donut chart
            )
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                height=320,
                showlegend=True,
                legend=dict(orientation='h', y=-0.1)
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>'
                              'Percentage: %{percent}<extra></extra>'
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown(
            '<div class="section-header">Average CGPA by Category</div>',
            unsafe_allow_html=True
        )
        if df is not None:
            order   = ['At Risk', 'Average', 'Good', 'Excellent']
            avg_cgpa_cat = (
                df.groupby('performance_category')['previous_cgpa']
                .mean()
                .reindex(order)
                .reset_index()
            )
            fig2 = px.bar(
                avg_cgpa_cat,
                x='performance_category',
                y='previous_cgpa',
                color='performance_category',
                color_discrete_map=colors,
                text='previous_cgpa',
                labels={
                    'performance_category': 'Category',
                    'previous_cgpa': 'Avg CGPA'
                }
            )
            fig2.update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside'
            )
            fig2.update_layout(
                showlegend=False,
                height=320,
                margin=dict(t=20, b=0),
                yaxis=dict(range=[0, 11])
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── How it works section ─────────────────────────────────
    st.markdown(
        '<div class="section-header">How It Works</div>',
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("📥", "Input Data",
         "Enter student academic metrics including attendance, "
         "study hours, and marks"),
        ("🤖", "ML Prediction",
         "Random Forest model analyses the inputs and predicts "
         "one of 4 performance categories"),
        ("📊", "Confidence Score",
         "The model provides a probability score showing how "
         "confident it is in the prediction"),
        ("💡", "Recommendations",
         "Personalised, prioritised action plan is generated "
         "based on identified weak areas"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div style='background:white; padding:20px; border-radius:12px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06);
                        text-align:center; height:180px;'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-weight:700; color:#2c3e50;
                            margin:8px 0 6px;'>{title}</div>
                <div style='font-size:0.85rem; color:#7f8c8d;
                            line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2: PREDICT PERFORMANCE
# ─────────────────────────────────────────────

LABEL_MAP = {0: 'At Risk', 1: 'Average', 2: 'Good', 3: 'Excellent'}

def render_prediction_page(model, feature_cols):

    st.markdown(
        "<h2 style='color:#2c3e50;'>🔮 Student Performance Predictor</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "Fill in the student details below and click **Predict** "
        "to get an instant performance prediction with personalised "
        "recommendations."
    )
    st.markdown("---")

    # ── INPUT FORM ────────────────────────────────────────────
    with st.form("prediction_form"):

        st.markdown(
            '<div class="section-header">📋 Academic Information</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            attendance = st.slider(
                "📅 Attendance Percentage",
                min_value=0.0, max_value=100.0,
                value=75.0, step=0.5,
                help="Percentage of classes attended this semester"
            )
            study_hours = st.slider(
                "📚 Study Hours per Day",
                min_value=0.0, max_value=12.0,
                value=4.0, step=0.5,
                help="Average number of hours studied per day"
            )
            assignment_rate = st.slider(
                "📝 Assignment Completion Rate (%)",
                min_value=0.0, max_value=100.0,
                value=80.0, step=1.0,
                help="Percentage of assignments submitted on time"
            )

        with col2:
            internal_marks = st.slider(
                "📊 Internal Marks (out of 100)",
                min_value=0.0, max_value=100.0,
                value=60.0, step=0.5,
                help="Average of all internal/unit test marks"
            )
            previous_cgpa = st.slider(
                "🎓 Previous Semester CGPA",
                min_value=0.0, max_value=10.0,
                value=6.5, step=0.1,
                help="CGPA from the previous semester (0–10 scale)"
            )
            age = st.number_input(
                "🧑 Age",
                min_value=17, max_value=30,
                value=20, step=1
            )

        st.markdown(
            '<div class="section-header">👤 Personal Information</div>',
            unsafe_allow_html=True
        )

        col3, col4 = st.columns(2)

        with col3:
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female", "Other"]
            )
            extracurricular = st.radio(
                "🎭 Extracurricular Activities",
                options=["Yes", "No"],
                horizontal=True,
                help="Does the student participate in clubs, sports, etc?"
            )

        with col4:
            part_time_job = st.radio(
                "💼 Part-time Job",
                options=["No", "Yes"],
                horizontal=True,
                help="Does the student have a part-time job?"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "🚀 Predict Performance",
            use_container_width=True,
            type="primary"
        )

    # ── PREDICTION LOGIC ──────────────────────────────────────
    if submitted:
        if model is None:
            st.error(
                "Model not found. Please run "
                "`python src\\model_training.py` first."
            )
            return

        # Build the student data dictionary
        student_data = {
            'attendance_percentage'      : attendance,
            'study_hours_per_day'        : study_hours,
            'assignment_completion_rate' : assignment_rate,
            'internal_marks'             : internal_marks,
            'previous_cgpa'              : previous_cgpa,
            'extracurricular_activities' : 1 if extracurricular == "Yes" else 0,
            'part_time_job'              : 1 if part_time_job == "Yes" else 0,
        }

        # Feature engineering (same as preprocessing)
        student_data['academic_score_index'] = (
            (attendance * 0.25) +
            (internal_marks * 0.35) +
            (assignment_rate * 0.20) +
            (study_hours / 12 * 100 * 0.20)
        )
        student_data['engagement_score'] = (
            (assignment_rate * 0.5) + (attendance * 0.5)
        )
        student_data['study_efficiency'] = (
            previous_cgpa / (study_hours + 0.1)
        )

        # Gender encoding (match training data format)
        student_data['gender_Male']  = 1 if gender == "Male"  else 0
        student_data['gender_Other'] = 1 if gender == "Other" else 0

        # Build input array in the exact same order as training
        input_values = []
        for col in feature_cols:
            input_values.append(student_data.get(col, 0))

        input_array = np.array(input_values).reshape(1, -1)

        # Make prediction
        prediction_encoded = model.predict(input_array)[0]
        prediction_proba   = model.predict_proba(input_array)[0]
        predicted_category = LABEL_MAP[prediction_encoded]
        confidence         = prediction_proba[prediction_encoded] * 100

        # Show result
        _display_prediction_result(
            predicted_category,
            confidence,
            prediction_proba,
            student_data
        )


def _display_prediction_result(category, confidence,
                                probabilities, student_data):
    """Renders the prediction result and recommendations."""

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#2c3e50;'>📊 Prediction Results</h3>",
        unsafe_allow_html=True
    )

    # ── Result card ───────────────────────────────────────────
    css_class = {
        'Excellent' : 'result-excellent',
        'Good'      : 'result-good',
        'Average'   : 'result-average',
        'At Risk'   : 'result-at-risk'
    }[category]

    emoji = {
        'Excellent' : '🌟',
        'Good'      : '👍',
        'Average'   : '📈',
        'At Risk'   : '🆘'
    }[category]

    st.markdown(f"""
    <div class="{css_class}">
        <h2 style='margin:0; font-size:2rem;'>
            {emoji} Predicted: {category}
        </h2>
        <p style='margin:8px 0 0; font-size:1.1rem; opacity:0.85;'>
            Model confidence: <strong>{confidence:.1f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence chart ──────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Prediction Probability Breakdown**")
        labels  = ['At Risk', 'Average', 'Good', 'Excellent']
        colors  = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        fig = go.Figure(go.Bar(
            x=labels,
            y=[p * 100 for p in probabilities],
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='outside'
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=30, b=0, l=0, r=0),
            yaxis=dict(range=[0, 115], title='Probability (%)'),
            xaxis=dict(title='Category'),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Student Input Summary**")
        metrics = {
            "Attendance"        : f"{student_data['attendance_percentage']:.1f}%",
            "Study Hours/Day"   : f"{student_data['study_hours_per_day']:.1f} hrs",
            "Assignment Rate"   : f"{student_data['assignment_completion_rate']:.1f}%",
            "Internal Marks"    : f"{student_data['internal_marks']:.1f}/100",
            "Previous CGPA"     : f"{student_data['previous_cgpa']:.2f}/10",
            "Academic Index"    : f"{student_data['academic_score_index']:.1f}",
        }
        for label, value in metrics.items():
            col_a, col_b = st.columns([2, 1])
            col_a.write(f"**{label}**")
            col_b.write(value)

    # ── Recommendations ───────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<h3 style='color:#2c3e50;'>💡 Personalised Recommendations</h3>",
        unsafe_allow_html=True
    )

    report = generate_recommendations(student_data, category)

    # Motivational message
    cat_info = report['category_info']
    st.info(f"{cat_info['emoji']} **{cat_info['title']}**\n\n"
            f"{cat_info['message']}")

    # Priority action plan
    if report['improvement_plan']:
        st.markdown("#### 🎯 Your Priority Action Plan")

        for step in report['improvement_plan']:
            status = report['recommendations'].get(
                step['metric'].lower().replace(' ', '_'), {}
            ).get('status', 'low')

            border_color = {
                'critical': '#e74c3c',
                'low'     : '#f39c12',
                'good'    : '#2ecc71'
            }.get(status, '#3498db')

            with st.expander(
                f"Priority #{step['priority']}: "
                f"{step['metric']} — "
                f"Current: {step['current']:.1f} → "
                f"Target: {step['target']}",
                expanded=(step['priority'] == 1)
            ):
                for tip in [step['top_tip']] + step['detail_tips']:
                    st.markdown(f"- {tip}")

    # Benchmark comparison
    st.markdown("#### 📏 How You Compare to the Excellent Benchmark")

    benchmark_data = []
    for metric, gap_info in report['benchmark_gap'].items():
        clean = metric.replace('_', ' ').title()
        benchmark_data.append({
            'Metric'     : clean,
            'Your Value' : gap_info['student_value'],
            'Target'     : gap_info['benchmark_value'],
            'Gap'        : gap_info['gap']
        })

    bdf = pd.DataFrame(benchmark_data)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Your Score',
        x=bdf['Metric'],
        y=bdf['Your Value'],
        marker_color='#3498db',
        text=bdf['Your Value'].round(1),
        textposition='auto'
    ))
    fig3.add_trace(go.Bar(
        name='Excellent Benchmark',
        x=bdf['Metric'],
        y=bdf['Target'],
        marker_color='#2ecc71',
        opacity=0.6,
        text=bdf['Target'],
        textposition='auto'
    ))
    fig3.update_layout(
        barmode='group',
        height=350,
        margin=dict(t=20, b=0),
        legend=dict(orientation='h', y=1.1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Lifestyle tips
    if report['lifestyle_tips']:
        st.markdown("#### 🌱 Lifestyle & Balance Tips")
        for tip in report['lifestyle_tips']:
            st.markdown(f"- {tip}")


# ─────────────────────────────────────────────
# PAGE 3: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────

def render_analytics(df):
    st.markdown(
        "<h2 style='color:#2c3e50;'>📊 Analytics Dashboard</h2>",
        unsafe_allow_html=True
    )

    if df is None:
        st.error("Dataset not found.")
        return

    colors = {
        'Excellent': '#2ecc71',
        'Good'     : '#3498db',
        'Average'  : '#f39c12',
        'At Risk'  : '#e74c3c'
    }
    order = ['At Risk', 'Average', 'Good', 'Excellent']

    tab1, tab2, tab3 = st.tabs([
        "📈 Distributions",
        "🔗 Correlations",
        "🎯 Feature Insights"
    ])

    # ── Tab 1: Distributions ─────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x='attendance_percentage',
                color='performance_category',
                color_discrete_map=colors,
                nbins=30,
                title='Attendance Distribution by Category',
                labels={'attendance_percentage': 'Attendance (%)'},
                barmode='overlay',
                opacity=0.7
            )
            fig.update_layout(height=350, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df, x='study_hours_per_day',
                color='performance_category',
                color_discrete_map=colors,
                nbins=25,
                title='Study Hours Distribution by Category',
                labels={'study_hours_per_day': 'Study Hours/Day'},
                barmode='overlay',
                opacity=0.7
            )
            fig.update_layout(height=350, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.box(
                df,
                x='performance_category',
                y='previous_cgpa',
                color='performance_category',
                color_discrete_map=colors,
                category_orders={'performance_category': order},
                title='CGPA Distribution by Category',
                labels={
                    'previous_cgpa': 'Previous CGPA',
                    'performance_category': 'Category'
                }
            )
            fig.update_layout(
                height=350, margin=dict(t=40,b=0),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.box(
                df,
                x='performance_category',
                y='internal_marks',
                color='performance_category',
                color_discrete_map=colors,
                category_orders={'performance_category': order},
                title='Internal Marks by Category',
                labels={
                    'internal_marks': 'Internal Marks',
                    'performance_category': 'Category'
                }
            )
            fig.update_layout(
                height=350, margin=dict(t=40,b=0),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Correlations ───────────────────────────────────
    with tab2:
        numerical_cols = [
            'attendance_percentage', 'study_hours_per_day',
            'assignment_completion_rate', 'internal_marks',
            'previous_cgpa', 'academic_score_index',
            'engagement_score', 'performance_encoded'
        ]
        existing_cols = [c for c in numerical_cols if c in df.columns]
        corr          = df[existing_cols].corr().round(3)

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdYlGn',
            zmin=-1, zmax=1,
            title='Feature Correlation Heatmap',
            aspect='auto'
        )
        fig.update_layout(height=550, margin=dict(t=50,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Correlation with target
        st.markdown("**Correlation with Performance Score (sorted)**")
        target_corr = (
            df[existing_cols]
            .corr()['performance_encoded']
            .drop('performance_encoded')
            .sort_values(ascending=False)
        )
        fig2 = px.bar(
            x=target_corr.index,
            y=target_corr.values,
            color=target_corr.values,
            color_continuous_scale='RdYlGn',
            labels={'x': 'Feature', 'y': 'Correlation'},
            title='Feature Correlation with Performance'
        )
        fig2.update_layout(height=350, margin=dict(t=40,b=0),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Feature Insights ───────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df.dropna(subset=['study_hours_per_day',
                                  'previous_cgpa',
                                  'internal_marks']),
                x='study_hours_per_day',
                y='previous_cgpa',
                color='performance_category',
                color_discrete_map=colors,
                size='internal_marks',
                size_max=14,
                opacity=0.65,
                title='Study Hours vs CGPA (size = internal marks)',
                labels={
                    'study_hours_per_day': 'Study Hours/Day',
                    'previous_cgpa'      : 'Previous CGPA'
                }
            )
            fig.update_layout(height=400, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            avg_metrics = (
                df.groupby('performance_category')[[
                    'attendance_percentage',
                    'study_hours_per_day',
                    'assignment_completion_rate',
                    'internal_marks'
                ]].mean()
                .reindex(order)
                .reset_index()
            )
            fig = px.bar(
                avg_metrics.melt(
                    id_vars='performance_category',
                    var_name='Metric',
                    value_name='Average Value'
                ),
                x='Metric', y='Average Value',
                color='performance_category',
                color_discrete_map=colors,
                barmode='group',
                title='Average Metrics by Performance Category',
                category_orders={'performance_category': order}
            )
            fig.update_layout(height=400, margin=dict(t=40,b=0),
                              xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 4: DATASET EXPLORER
# ─────────────────────────────────────────────

def render_dataset(df):
    st.markdown(
        "<h2 style='color:#2c3e50;'>📋 Dataset Explorer</h2>",
        unsafe_allow_html=True
    )

    if df is None:
        st.error("Dataset not found.")
        return

    # Filter controls
    st.markdown("**Filter Dataset**")
    col1, col2, col3 = st.columns(3)

    with col1:
        category_filter = st.multiselect(
            "Performance Category",
            options=['Excellent', 'Good', 'Average', 'At Risk'],
            default=['Excellent', 'Good', 'Average', 'At Risk']
        )
    with col2:
        cgpa_range = st.slider(
            "CGPA Range",
            min_value=0.0, max_value=10.0,
            value=(0.0, 10.0), step=0.1
        )
    with col3:
        attendance_range = st.slider(
            "Attendance Range (%)",
            min_value=0.0, max_value=100.0,
            value=(0.0, 100.0), step=1.0
        )

    # Apply filters
    filtered_df = df[
        (df['performance_category'].isin(category_filter)) &
        (df['previous_cgpa'].between(*cgpa_range)) &
        (df['attendance_percentage'].between(*attendance_range))
    ]

    st.markdown(
        f"Showing **{len(filtered_df)}** of **{len(df)}** students"
    )

    # Display columns
    display_cols = [
        'student_id', 'age', 'attendance_percentage',
        'study_hours_per_day', 'assignment_completion_rate',
        'internal_marks', 'previous_cgpa',
        'performance_category'
    ]
    existing_display = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[existing_display].reset_index(drop=True),
        use_container_width=True,
        height=420
    )

    # Summary stats
    st.markdown("**Summary Statistics**")
    num_cols = [
        'attendance_percentage', 'study_hours_per_day',
        'internal_marks', 'previous_cgpa'
    ]
    existing_num = [c for c in num_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[existing_num].describe().round(2),
        use_container_width=True
    )


# ─────────────────────────────────────────────
# MAIN APP ENTRY POINT
# ─────────────────────────────────────────────

def main():
    # Load resources
    model       = load_model()
    feature_cols = load_feature_cols()
    df          = load_dataset()

    # Render navigation
    page = render_sidebar()

    # Route to correct page
    if   page == "🏠 Home":
        render_home(df)
    elif page == "🔮 Predict Performance":
        render_prediction_page(model, feature_cols)
    elif page == "📊 Analytics Dashboard":
        render_analytics(df)
    elif page == "📋 Dataset Explorer":
        render_dataset(df)


if __name__ == "__main__":
    main()