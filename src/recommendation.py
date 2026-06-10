# =============================================================
# FILE: src/recommendation.py
# PURPOSE: Generate personalised recommendations based on
#          student performance prediction and input data
# =============================================================

# ─────────────────────────────────────────────
# SECTION 1: THRESHOLDS AND BENCHMARK DEFINITIONS
# ─────────────────────────────────────────────

# These thresholds define what "good" looks like for each metric.
# Based on typical engineering college standards in India.
# Every threshold is a documented, explainable decision —
# important for interviews.

THRESHOLDS = {
    'attendance_percentage': {
        'critical': 60,    # Below this → immediate concern
        'low': 75,    # Below this → needs improvement
        'good': 85,    # Above this → healthy
    },
    'study_hours_per_day': {
        'critical': 2,
        'low': 3,
        'good': 5,
    },
    'assignment_completion_rate': {
        'critical': 60,
        'low': 75,
        'good': 90,
    },
    'internal_marks': {
        'critical': 40,
        'low': 55,
        'good': 70,
    },
    'previous_cgpa': {
        'critical': 5.0,
        'low': 6.5,
        'good': 8.0,
    },
}

# Benchmark: What does an "Excellent" student look like?
# Used to show students how far they are from the top tier.
EXCELLENT_BENCHMARK = {
    'attendance_percentage': 90,
    'study_hours_per_day': 6,
    'assignment_completion_rate': 95,
    'internal_marks': 80,
    'previous_cgpa': 8.5,
}

# ─────────────────────────────────────────────
# SECTION 2: RECOMMENDATION LIBRARY
# ─────────────────────────────────────────────
# Each metric has 3 levels of recommendations:
# critical, low, and good (encouragement)

RECOMMENDATION_LIBRARY = {

    'attendance_percentage': {
        'critical': [
            "🚨 URGENT: Your attendance is critically low. "
            "Most universities require 75% minimum to sit exams.",
            "📋 Action: Contact your academic advisor immediately "
            "to discuss attendance improvement plan.",
            "📅 Strategy: Use a daily alarm and a physical attendance "
            "tracker. Treat every class like a job interview.",
            "👥 Tip: Find a classmate accountability partner who will "
            "text you every morning before class.",
        ],
        'low': [
            "⚠️ Your attendance is below the recommended 75% threshold.",
            "📅 Strategy: Set phone reminders 30 minutes before each "
            "class. Prepare the night before to avoid morning rush.",
            "📊 Goal: Aim to attend every remaining class this semester. "
            "Even improving to 80% will significantly boost your grade.",
            "💡 Insight: Studies show that attendance is the single "
            "strongest predictor of academic performance.",
        ],
        'good': [
            "✅ Great attendance! You are showing up consistently.",
            "🎯 Challenge: Can you reach 95%? Perfect attendance often "
            "earns extra credit and impresses professors.",
        ],
    },

    'study_hours_per_day': {
        'critical': [
            "🚨 You are studying less than 2 hours per day — "
            "this is insufficient for engineering coursework.",
            "📚 Start small: Add just 30 minutes of focused study "
            "today. Build the habit before increasing duration.",
            "⏱️ Technique: Use the Pomodoro method — 25 minutes of "
            "focused study, 5-minute break. No phone during study.",
            "🌙 Schedule: Study at the same time every day to build "
            "a habit. Early morning (6–8 AM) is most effective.",
        ],
        'low': [
            "⚠️ Aim for at least 4–5 focused study hours per day "
            "for solid academic performance.",
            "📖 Quality over quantity: 3 hours of deep, distraction-free "
            "study beats 6 hours of passive reading.",
            "🗓️ Weekly plan: Allocate specific subjects to specific days. "
            "Avoid leaving everything to the day before exams.",
            "📵 Remove distractions: Put your phone in another room, "
            "use apps like Forest or Cold Turkey to block social media.",
        ],
        'good': [
            "✅ Good study discipline! Keep up the consistent effort.",
            "📈 Optimise: Ensure your study time includes active recall "
            "(practice problems, flashcards) not just passive reading.",
        ],
    },

    'assignment_completion_rate': {
        'critical': [
            "🚨 CRITICAL: Missing assignments severely impacts your "
            "internal marks and overall CGPA.",
            "📝 Immediate action: Submit all pending assignments, even "
            "if late. Partial credit is better than zero.",
            "🗂️ System: Create a dedicated assignment tracker. Use "
            "Google Calendar or a physical planner for all deadlines.",
            "🤝 Strategy: Form a study group. Discussing assignments "
            "with peers helps you understand and complete them faster.",
        ],
        'low': [
            "⚠️ Your assignment completion is below 75%. "
            "This directly reduces your internal marks.",
            "📋 Habit: Start every assignment on the day it is given, "
            "even for just 10 minutes. Starting is the hardest part.",
            "⏰ Rule: Never miss two assignments in a row. "
            "One miss is an exception; two is a pattern.",
        ],
        'good': [
            "✅ Strong assignment completion. This directly supports "
            "your internal marks and shows consistency.",
            "🎯 Push for 100%: Every assignment completed on time "
            "builds the professional habits employers look for.",
        ],
    },

    'internal_marks': {
        'critical': [
            "🚨 Your internal marks need immediate attention. "
            "These scores directly determine your final CGPA.",
            "📖 Strategy: Focus on the most recent 2 unit tests. "
            "Understand what topics you lost marks on specifically.",
            "👨‍🏫 Action: Visit your subject professors during office "
            "hours. Ask exactly which topics carry the most weight.",
            "📝 Revision technique: Solve previous years' question "
            "papers. Pattern recognition is key to exam success.",
        ],
        'low': [
            "⚠️ Your internal marks are below average. "
            "Aim for at least 60/100 to maintain a healthy CGPA.",
            "🧠 Active recall: After studying a topic, close the book "
            "and write down everything you remember. Test yourself.",
            "📊 Track: Identify your weakest subject and spend "
            "double time on it this week.",
        ],
        'good': [
            "✅ Your internal marks are solid. Well done.",
            "🚀 Next level: Aim for distinction (75+). "
            "It qualifies you for academic scholarships and honours.",
        ],
    },

    'previous_cgpa': {
        'critical': [
            "🚨 Your CGPA is in the critical range. Many companies "
            "have a 6.0 CGPA cutoff for campus placements.",
            "🎯 Target: Focus on this semester entirely. "
            "A strong semester can raise your CGPA by 0.3–0.5 points.",
            "📋 Academic plan: Meet with your academic counsellor and "
            "create a formal improvement plan this week.",
            "💼 Backup strategy: Build strong projects and internship "
            "experience to compensate for CGPA in interviews.",
        ],
        'low': [
            "⚠️ Your CGPA is below 6.5. Many top recruiters "
            "filter at 6.0–7.0 during campus placements.",
            "📈 Each semester matters: Scoring 8+ this semester "
            "will meaningfully raise your cumulative CGPA.",
            "🔍 Know your cutoffs: Research the CGPA requirements "
            "of companies you want to apply to and target accordingly.",
        ],
        'good': [
            "✅ Your CGPA is good. You meet most placement cutoffs.",
            "🌟 Chase 9+: A CGPA above 9 opens doors to top-tier "
            "companies, higher studies, and merit scholarships.",
        ],
    },
}

# ─────────────────────────────────────────────
# SECTION 3: LIFESTYLE RECOMMENDATIONS
# Based on extracurricular and part-time job status
# ─────────────────────────────────────────────

LIFESTYLE_RECOMMENDATIONS = {
    'extracurricular_no': [
        "🎯 Consider joining one technical club or society. "
        "Hackathons and coding clubs add real value to your resume.",
        "🤝 Networking: College events connect you with seniors who "
        "have already been placed — their advice is invaluable.",
    ],
    'extracurricular_yes': [
        "✅ Good — extracurriculars show well-rounded development.",
        "⚖️ Balance check: Ensure activities don't cut into your "
        "study or sleep time. Academics remain the priority.",
    ],
    'part_time_job_risk': [
        "⚠️ Your part-time job may be affecting your study time. "
        "Consider reducing hours during exam weeks.",
        "🧮 Calculate: If work + commute > 3 hours/day, it likely "
        "impacts your academic performance significantly.",
    ],
    'part_time_job_ok': [
        "✅ Managing work and studies shows great time management — "
        "a skill that impresses employers.",
    ],
}

# ─────────────────────────────────────────────
# SECTION 4: CATEGORY-LEVEL MOTIVATIONAL MESSAGE
# ─────────────────────────────────────────────

CATEGORY_MESSAGES = {
    'Excellent': {
        'emoji': '🌟',
        'title': 'Outstanding Performance!',
        'message': (
            "You are in the top tier of your class. "
            "Your consistent effort and discipline are paying off. "
            "Focus on maintaining this momentum and exploring "
            "advanced opportunities like research, internships, "
            "and competitive exams."
        ),
        'color': '#2ecc71',
    },
    'Good': {
        'emoji': '👍',
        'title': 'Good Performance — Keep Pushing!',
        'message': (
            "You are performing well above average. "
            "With focused effort on your weak areas, "
            "you have a real chance of reaching the Excellent tier "
            "this semester. You are closer than you think."
        ),
        'color': '#3498db',
    },
    'Average': {
        'emoji': '📈',
        'title': 'Average Performance — Room to Grow',
        'message': (
            "Your performance is at the average level. "
            "This is a turning point — small, consistent improvements "
            "in attendance and study hours can move you to Good within "
            "one semester. The gap is bridgeable."
        ),
        'color': '#f39c12',
    },
    'At Risk': {
        'emoji': '🆘',
        'title': 'At Risk — Immediate Action Needed',
        'message': (
            "Your current performance puts you at risk of failing "
            "or missing placement cutoffs. But this is recoverable. "
            "Focus on the top 2 recommendations below, "
            "speak to your academic advisor today, "
            "and commit to one week of maximum effort."
        ),
        'color': '#e74c3c',
    },
}

# ─────────────────────────────────────────────
# SECTION 5: CORE RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────


def generate_recommendations(student_data: dict, predicted_category: str) -> dict:
    """
    Main function: generates a complete, personalised
    recommendation report for a single student.

    Parameters:
        student_data (dict): Student's input values
            Keys: attendance_percentage, study_hours_per_day,
                  assignment_completion_rate, internal_marks,
                  previous_cgpa, extracurricular_activities,
                  part_time_job

        predicted_category (str): One of
            'Excellent', 'Good', 'Average', 'At Risk'

    Returns:
        dict: {
            'category_info'      : motivation message and color,
            'weak_areas'         : list of identified weak metrics,
            'recommendations'    : dict of tips per metric,
            'lifestyle_tips'     : extracurricular/job advice,
            'improvement_plan'   : prioritised 3-step action plan,
            'benchmark_gap'      : how far from Excellent benchmark,
        }
    """

    recommendations = {}
    weak_areas = []
    severity_scores = {}   # Used to prioritise the action plan

    # ── Step 1: Diagnose each metric ─────────────────────────
    academic_metrics = [
        'attendance_percentage',
        'study_hours_per_day',
        'assignment_completion_rate',
        'internal_marks',
        'previous_cgpa',
    ]

    for metric in academic_metrics:
        value = student_data.get(metric, None)
        if value is None:
            continue

        thresholds = THRESHOLDS[metric]
        tip_bank = RECOMMENDATION_LIBRARY[metric]

        if value < thresholds['critical']:
            # Critical weakness — highest priority
            recommendations[metric] = {
                'status': 'critical',
                'tips': tip_bank['critical'],
                'value': value,
                'target': thresholds['good'],
            }
            weak_areas.append(metric)
            # Severity score: how far below critical threshold
            severity_scores[metric] = (thresholds['critical'] - value) * 3

        elif value < thresholds['low']:
            # Moderate weakness
            recommendations[metric] = {
                'status': 'low',
                'tips': tip_bank['low'],
                'value': value,
                'target': thresholds['good'],
            }
            weak_areas.append(metric)
            severity_scores[metric] = thresholds['low'] - value

        else:
            # Doing well — give positive reinforcement
            recommendations[metric] = {
                'status': 'good',
                'tips': tip_bank['good'],
                'value': value,
                'target': thresholds['good'],
            }

    # ── Step 2: Lifestyle recommendations ────────────────────
    lifestyle_tips = []

    extra = student_data.get('extracurricular_activities', 0)
    if extra == 0:
        lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS['extracurricular_no'])
    else:
        lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS['extracurricular_yes'])

    job = student_data.get('part_time_job', 0)
    study_hours = student_data.get('study_hours_per_day', 5)
    if job == 1 and study_hours < 4:
        # Has a job AND studies less than 4 hours — flag the conflict
        lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS['part_time_job_risk'])
    elif job == 1:
        lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS['part_time_job_ok'])

    # ── Step 3: Build prioritised improvement plan ───────────
    # Sort weak areas by severity (most critical first)
    sorted_weak = sorted(
        severity_scores,
        key=lambda k: severity_scores[k],
        reverse=True
    )

    # Give the student maximum 3 focus areas (cognitive overload prevention)
    improvement_plan = []
    for i, metric in enumerate(sorted_weak[:3], start=1):
        rec = recommendations[metric]
        clean = metric.replace('_', ' ').title()
        improvement_plan.append({
            'priority': i,
            'metric': clean,
            'current': rec['value'],
            'target': rec['target'],
            'top_tip': rec['tips'][0],     # Most important tip only
            'detail_tips': rec['tips'][1:],    # Supporting tips
        })

    # ── Step 4: Calculate gap from Excellent benchmark ───────
    benchmark_gap = {}
    for metric, benchmark_val in EXCELLENT_BENCHMARK.items():
        student_val = student_data.get(metric, 0)
        gap = benchmark_val - student_val
        benchmark_gap[metric] = {
            'student_value': student_val,
            'benchmark_value': benchmark_val,
            'gap': round(gap, 2),
            'gap_pct': round((gap / benchmark_val) * 100, 1)
            if benchmark_val != 0 else 0,
        }

    return {
        'category_info': CATEGORY_MESSAGES[predicted_category],
        'predicted_category': predicted_category,
        'weak_areas': weak_areas,
        'recommendations': recommendations,
        'lifestyle_tips': lifestyle_tips,
        'improvement_plan': improvement_plan,
        'benchmark_gap': benchmark_gap,
    }


# ─────────────────────────────────────────────
# SECTION 6: DISPLAY FUNCTION (Terminal output)
# Used for testing — dashboard will use its own UI
# ─────────────────────────────────────────────

def display_recommendations(report: dict):
    """
    Prints a formatted recommendation report to terminal.
    Useful for testing the engine without the dashboard.
    """
    cat_info = report['category_info']

    print("\n" + "=" * 60)
    print(f"  {cat_info['emoji']}  {cat_info['title']}")
    print("=" * 60)
    print(f"\n{cat_info['message']}\n")

    if report['improvement_plan']:
        print("─" * 60)
        print("  YOUR TOP PRIORITY ACTION PLAN")
        print("─" * 60)

        for step in report['improvement_plan']:
            print(f"\n  Priority #{step['priority']}: "
                  f"{step['metric']}")
            print(f"  Current : {step['current']:.1f}  →  "
                  f"Target: {step['target']}")
            print(f"  {step['top_tip']}")
            for tip in step['detail_tips']:
                print(f"  {tip}")

    if report['lifestyle_tips']:
        print("\n" + "─" * 60)
        print("  LIFESTYLE & BALANCE TIPS")
        print("─" * 60)
        for tip in report['lifestyle_tips']:
            print(f"  {tip}")

    print("\n" + "─" * 60)
    print("  HOW FAR YOU ARE FROM EXCELLENT")
    print("─" * 60)
    for metric, gap_info in report['benchmark_gap'].items():
        clean = metric.replace('_', ' ').title()
        gap = gap_info['gap']
        status = "✅ Above target" if gap <= 0 else f"↑ Need +{gap:.1f}"
        print(f"  {clean:<30} {gap_info['student_value']:.1f}"
              f"  {status}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────
# SECTION 7: TEST THE ENGINE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Test Case 1: An At Risk student
    print("\n>>> TEST CASE 1: At Risk Student")
    at_risk_student = {
        'attendance_percentage': 52.0,
        'study_hours_per_day': 1.5,
        'assignment_completion_rate': 55.0,
        'internal_marks': 35.0,
        'previous_cgpa': 4.8,
        'extracurricular_activities': 0,
        'part_time_job': 1,
    }
    report1 = generate_recommendations(at_risk_student, 'At Risk')
    display_recommendations(report1)

    # Test Case 2: A Good student
    print("\n>>> TEST CASE 2: Good Student")
    good_student = {
        'attendance_percentage': 80.0,
        'study_hours_per_day': 4.5,
        'assignment_completion_rate': 85.0,
        'internal_marks': 65.0,
        'previous_cgpa': 7.2,
        'extracurricular_activities': 1,
        'part_time_job': 0,
    }
    report2 = generate_recommendations(good_student, 'Good')
    display_recommendations(report2)

    # Test Case 3: An Excellent student
    print("\n>>> TEST CASE 3: Excellent Student")
    excellent_student = {
        'attendance_percentage': 92.0,
        'study_hours_per_day': 6.5,
        'assignment_completion_rate': 97.0,
        'internal_marks': 85.0,
        'previous_cgpa': 9.1,
        'extracurricular_activities': 1,
        'part_time_job': 0,
    }
    report3 = generate_recommendations(excellent_student, 'Excellent')
    display_recommendations(report3)
