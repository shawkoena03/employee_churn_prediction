# =============================================================
# PHASE 6 — INTERACTIVE RETENTION DASHBOARD
# Project : Employee Churn Prediction
# Streamlit Web App
# =============================================================

import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
# pyrefly: ignore [missing-import]
import shap
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import textwrap

# Import custom email and agentic modules dynamically (since filenames start with digits)
import importlib
notification_module = importlib.import_module("5_notification")
send_retention_email = notification_module.send_retention_email
SENDER_EMAIL = notification_module.SENDER_EMAIL
HR_EMAIL = notification_module.HR_EMAIL

agent_module = importlib.import_module("4_agentic_logic")
get_agentic_recommendations_level3 = agent_module.get_agentic_recommendations_level3

# =============================================================
# APP CONFIGURATION & STYLING
# =============================================================
st.set_page_config(
    page_title="Saudi Attrition & Retention Intelligence Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for modern card layouts and colors
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        /* Reset and apply Modern Typography */
        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', sans-serif !important;
            background: linear-gradient(135deg, #020617 0%, #0f172a 100%) !important;
            color: #f8fafc !important;
        }
        
        /* Main heading style */
        h1 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
            background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe, #00f2fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(79, 172, 254, 0.2);
        }
        
        /* Sidebar styling - Glassmorphic */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.6) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        [data-testid="stSidebar"] [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #38bdf8 !important;
            font-size: 14px;
            font-weight: 700;
            margin-top: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        /* Key statistics metric cards - Glassmorphism */
        .metric-card {
            background: rgba(30, 41, 59, 0.4) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px !important;
            padding: 24px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
            text-align: left !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 15px 40px 0 rgba(0, 0, 0, 0.4), 0 0 20px rgba(56, 189, 248, 0.2) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }
        .metric-card:hover::before {
            opacity: 1;
        }
        
        .metric-value {
            font-size: 38px !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            margin-top: 6px !important;
            letter-spacing: -0.02em !important;
            line-height: 1.1 !important;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }
        
        .metric-label {
            font-size: 12px !important;
            color: #94a3b8 !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
        }
        
        /* Sparkline SVG styling */
        .sparkline-svg {
            margin-top: 14px;
            height: 35px;
            width: 100%;
            filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.6));
        }
        
        .card-header {
            font-weight: 800 !important;
            font-size: 20px !important;
            color: #f1f5f9 !important;
            border-bottom: none !important;
            padding-bottom: 0px !important;
            margin-top: 12px !important;
            margin-bottom: 20px !important;
            letter-spacing: -0.01em !important;
            display: flex;
            align-items: center;
        }
        
        /* Modern Driver Card Styling */
        .driver-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .driver-card:hover {
            transform: translateY(-3px) scale(1.01);
            background: rgba(30, 41, 59, 0.6);
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        
        .driver-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .driver-title {
            font-size: 16px;
            font-weight: 700;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .driver-impact {
            font-size: 14px;
            font-weight: 800;
            padding: 4px 12px;
            border-radius: 20px;
            background: rgba(0,0,0,0.2);
        }
        
        .driver-impact.negative {
            color: #ff4d4d;
            box-shadow: 0 0 10px rgba(255, 77, 77, 0.2);
            border: 1px solid rgba(255, 77, 77, 0.3);
        }
        
        .driver-impact.positive {
            color: #00e676;
            box-shadow: 0 0 10px rgba(0, 230, 118, 0.2);
            border: 1px solid rgba(0, 230, 118, 0.3);
        }
        
        .driver-desc {
            font-size: 13px;
            color: #cbd5e1;
            margin-top: 8px;
            line-height: 1.5;
        }
        
        .driver-icon-container {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            border-radius: 12px;
            font-size: 20px;
            font-weight: bold;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .driver-progress-bg {
            height: 6px;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 16px;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
        }
        
        .driver-progress-bar {
            height: 100%;
            border-radius: 999px;
            box-shadow: 0 0 10px currentColor;
        }
        
        /* Leaderboard styling */
        .leaderboard-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        
        .leaderboard-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 0px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            transition: background 0.2s;
        }
        .leaderboard-row:hover {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
        }
        .leaderboard-row:last-child {
            border-bottom: none;
            padding-bottom: 0px;
        }
        .leaderboard-row:first-child {
            padding-top: 0px;
        }
        
        .avatar-circle {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-weight: 800;
            font-size: 14px;
            color: #FFFFFF;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            border: 2px solid rgba(255,255,255,0.1);
        }
        
        .rank-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            font-weight: 800;
            color: #f1f5f9;
        }
        
        /* Intervention Card Styles (Tab 2 / 3) */
        .intervention-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .intervention-card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0,0,0,0.4), 0 0 15px rgba(168, 85, 247, 0.2);
            border-color: rgba(168, 85, 247, 0.4);
        }
        
        .intervention-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .intervention-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .intervention-badge-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            border-radius: 14px;
            font-size: 24px;
            background: rgba(255,255,255,0.05);
            box-shadow: inset 0 0 10px rgba(255,255,255,0.05);
        }
        
        .priority-badge {
            padding: 6px 12px;
            font-size: 11px;
            font-weight: 900;
            border-radius: 8px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        
        /* Adjust tab layouts */
        .stTabs [data-baseweb="tab-list"] {
            gap: 40px !important;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
            padding-bottom: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 700 !important;
            color: #94a3b8 !important;
            font-size: 16px !important;
            padding: 12px 20px !important;
            transition: all 0.3s ease !important;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #e2e8f0 !important;
            background: rgba(255,255,255,0.03);
        }
        .stTabs [aria-selected="true"] {
            color: #38bdf8 !important;
            border-bottom-color: #38bdf8 !important;
            text-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }
        
        /* Clean inputs & dropdowns styling */
        .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .stSelectbox div[data-baseweb="select"]:hover {
            border-color: #38bdf8 !important;
        }
        
        /* Checkbox overrides for dark mode */
        .stCheckbox p {
            color: #e2e8f0 !important;
            font-weight: 500 !important;
        }
        
        /* DataFrame overrides */
        [data-testid="stDataFrame"] {
            background: rgba(30, 41, 59, 0.4) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================
# LOAD PHASE 3 & 4 DATA
# =============================================================
OUTPUT_DIR = "outputs"

@st.cache_resource
def load_ml_resources():
    """Cache models and data splits for performance"""
    with open(os.path.join(OUTPUT_DIR, 'best_model_tuned.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(OUTPUT_DIR, 'test_data.pkl'), 'rb') as f:
        data = pickle.load(f)
        
    shap_importance = pd.read_csv(os.path.join(OUTPUT_DIR, 'shap_feature_importance.csv'))
    
    return model, data['X_test'], data['y_test'], data['X_train'], data['y_train'], shap_importance

try:
    model, X_test, y_test, X_train, y_train, shap_importance = load_ml_resources()
except Exception as e:
    st.error(f"Failed to load project outputs from Phase 3/4. Please make sure those scripts have been run first. (Error: {str(e)})")
    st.stop()

# Compute SHAP explainer
@st.cache_resource
def compute_shap_explainer(_model, _X_test):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_X_test)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    return explainer, sv

explainer, sv = compute_shap_explainer(model, X_test)

# =============================================================
# CONSTANTS & DICTIONARIES
# =============================================================
RISK_THRESHOLDS = {'CRITICAL': 0.80, 'HIGH': 0.60, 'MEDIUM': 0.40, 'LOW': 0.00}

FEATURE_LABELS = {
    'Job_Satisfaction': {0: 'Not Satisfied', 1: 'Satisfied', 2: 'Very Satisfied'},
    'Psychological_Exhaustion': {0: 'No', 1: 'Sometimes', 2: 'Yes'},
    'Physical_Stress': {0: 'No', 1: 'Sometimes', 2: 'Yes'},
    'Work_Live_Balance': {0: 'Easy', 1: 'Medium', 2: 'Difficult'},
    'MonthlySalary': {0: '1k–5k SAR', 1: '6k–10k SAR', 2: '11k–15k SAR', 3: '16k+ SAR'},
    'Get_Deserved_Promotion': {0: 'No', 1: 'Yes'},
    'Age': {0: '21–30', 1: '31–40', 2: '41+'},
    'Business_Travel': {0: 'No Travel', 1: 'Rarely', 2: 'Frequently'},
    'OverTime': {0: 'No', 1: 'Yes'},
    'Payment_Overtime': {0: 'No Overtime', 1: 'Overtime Not Paid', 2: 'Overtime Paid'},
    'Job_Opportunities': {0: 'No', 1: 'Yes'},
    'Job_Stability': {0: 'No', 1: 'Yes'},
    'Recognition': {0: 'No', 1: 'Yes'},
    'Bonus': {0: 'No', 1: 'Yes'},
    'MedicalInsurance': {0: 'No', 1: 'Yes'},
    'Rewards_Wages_Satisfaction': {0: 'No', 1: 'Yes'},
    'Useful_Training_Programs': {0: 'No', 1: 'Yes'},
    'Environment_Satisfaction': {0: 'Low', 1: 'Medium', 2: 'High'},
    'Emotional_Commitment': {0: 'Low', 1: 'Medium', 2: 'High'},
    'Job_Engagement': {0: 'Easy', 1: 'Medium', 2: 'Difficult'},
    'Job_Support': {0: 'Low', 1: 'Medium', 2: 'High'},
    'Distance_to_work': {0: 'Close', 1: 'Medium', 2: 'Far'},
    'Health_Issues': {0: 'No', 1: 'Yes'},
    'Academic_degree': {0: 'Diploma/Secondary', 1: 'Bachelor', 2: 'Master', 3: 'Ph.D.'},
    'Years_Experience': {0: '1–5 yrs', 1: '6–10 yrs', 2: '11+ yrs'},
    'Years_experience_lastorganization': {0: '1–5 yrs', 1: '6–10 yrs', 2: '11+ yrs'},
    'Training_programs_During_last_three_years': {0: 'No Trainings', 1: '1–3', 2: '4–6', 3: '7+'},
}

def classify_risk(probability: float) -> tuple:
    if probability >= RISK_THRESHOLDS['CRITICAL']:
        return ('CRITICAL', '🔴', '#D32F2F')
    elif probability >= RISK_THRESHOLDS['HIGH']:
        return ('HIGH', '🟠', '#F57C00')
    elif probability >= RISK_THRESHOLDS['MEDIUM']:
        return ('MEDIUM', '🟡', '#FBC02D')
    else:
        return ('LOW', '🟢', '#388E3C')

def get_employee_name(emp_id: int) -> str:
    names = [
        "Ahmad Al-Sudairy", "Sara Al-Otaibi", "Khalid Bin-Waleed", "Fatimah Al-Hassan",
        "Yousef Al-Harbi", "Muna Al-Subaie", "Tariq Al-Ghamdi", "Yasmin Al-Dosari",
        "Faisal Al-Qahtani", "Lulu Al-Saud", "Abdulrahman Al-Mutairi", "Reem Al-Anazi"
    ]
    return names[emp_id % len(names)]

def _get_action(feature: str, value) -> str:
    rules = {
        'Job_Satisfaction': {
            0: "Schedule 1-on-1 meeting to understand dissatisfaction. Review role alignment.",
            1: "Explore growth opportunities to elevate satisfaction to 'Very Satisfied'.",
        },
        'Psychological_Exhaustion': {
            1: "Assign a mentor. Review workload distribution across team.",
            2: "Immediate workload reduction required. Consider temporary reassignment or leave.",
        },
        'Physical_Stress': {
            1: "Review work environment and ergonomics. Consider flexible hours.",
            2: "Refer to occupational health. Evaluate role physical demands.",
        },
        'Work_Live_Balance': {
            1: "Introduce flexible scheduling options. Review meeting load.",
            2: "Enforce strict after-hours communication policy. Offer remote work options.",
        },
        'MonthlySalary': {
            0: "Conduct salary benchmarking. Employee is in lowest bracket — review for raise.",
            1: "Review compensation against market rate for role and experience.",
        },
        'Get_Deserved_Promotion': {
            0: "Review promotion eligibility. If overdue, initiate promotion discussion immediately.",
        },
        'OverTime': {
            1: "Audit overtime frequency. Redistribute tasks to reduce overreliance on this employee.",
        },
        'Payment_Overtime': {
            1: "Unpaid overtime is a high attrition driver. Establish overtime compensation policy.",
        },
        'Job_Opportunities': {
            1: "Employee perceives better opportunities elsewhere. Conduct retention conversation.",
        },
        'Job_Stability': {
            0: "Employee feels job is unstable. Provide reassurance and clarity on role security.",
        },
        'Recognition': {
            0: "Implement recognition program. Publicly acknowledge contributions in team settings.",
        },
        'Bonus': {
            0: "Review bonus eligibility. Performance-based incentives reduce attrition significantly.",
        },
        'Rewards_Wages_Satisfaction': {
            0: "Employee is dissatisfied with rewards/wages. Conduct compensation review.",
        },
        'Business_Travel': {
            2: "Frequent travel is a stress factor. Reduce travel frequency or offer travel allowances.",
        },
        'Distance_to_work': {
            2: "Long commute is a risk factor. Explore remote/hybrid work arrangements.",
        },
        'Health_Issues': {
            1: "Employee has health issues. Ensure access to medical insurance and flexible sick leave.",
        },
        'Environment_Satisfaction': {
            0: "Low environment satisfaction. Review workspace conditions and team dynamics.",
        },
        'Emotional_Commitment': {
            0: "Low emotional commitment. Assign meaningful projects aligned with employee's interests.",
        },
        'Job_Engagement': {
            2: "Employee finds job difficult to engage with. Review role fit and responsibilities.",
        },
        'MedicalInsurance': {
            0: "No medical insurance. This is a basic retention factor — enroll employee immediately.",
        },
        'Useful_Training_Programs': {
            0: "Training programs seen as not useful. Revamp with role-specific skill development.",
        },
        'Training_programs_During_last_three_years': {
            0: "No training in 3 years. Enroll in a structured development program immediately.",
        },
    }
    feature_rules = rules.get(feature, {})
    return feature_rules.get(value, None)

def get_recommendations(employee_series, shap_row, top_n=5):
    shap_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Value': employee_series.values,
        'SHAP': shap_row
    })
    risk_drivers = shap_df[shap_df['SHAP'] > 0].sort_values('SHAP', ascending=False).head(top_n)
    
    recommendations = []
    for _, row in risk_drivers.iterrows():
        feat = row['Feature']
        val = int(round(row['Value'])) if row['Value'] == round(row['Value']) else row['Value']
        shap_score = row['SHAP']
        label = FEATURE_LABELS.get(feat, {}).get(val, str(val))
        action = _get_action(feat, val)
        
        if action:
            recommendations.append({
                'feature': feat,
                'value_label': label,
                'shap_score': shap_score,
                'action': action,
                'priority': 'URGENT' if shap_score > 0.15 else 'RECOMMENDED'
            })
    return recommendations

# =============================================================
# COMPILE RESULTS DATAFRAME
# =============================================================
# Load classification threshold if optimized in Phase 3
threshold_path = os.path.join(OUTPUT_DIR, 'best_threshold.txt')
if os.path.exists(threshold_path):
    with open(threshold_path, 'r') as f:
        CLASSIFICATION_THRESHOLD = float(f.read().strip())
else:
    CLASSIFICATION_THRESHOLD = 0.50

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

results_list = []
for idx in range(len(X_test)):
    prob = probabilities[idx]
    level, emoji, color = classify_risk(prob)
    actual = y_test.iloc[idx]
    pred = predictions[idx]
    
    results_list.append({
        'Employee ID': X_test.index[idx],
        'Churn Probability': round(prob * 100, 1),
        'Risk Level': level,
        'Status': emoji + ' ' + level,
        'Predicted Outcome': 'Left' if pred == 1 else 'Stayed',
        'Actual Outcome': 'Left' if actual == 1 else 'Stayed',
        'Prediction Match': 'Match ✓' if pred == actual else 'Error ✗'
    })

results_df = pd.DataFrame(results_list)

# =============================================================
# INTERACTIVE SIDEBAR CONFIGURATION
# =============================================================
st.sidebar.markdown("""
    <div style="padding: 10px 0px 24px 0px; text-align: left; border-bottom: 2px solid #F1F5F9; margin-bottom: 20px;">
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 19px; font-weight: 800; color: #E11D48; letter-spacing: 0.12em; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
            <span>🇸🇦</span> SAUDI RETENTION
        </div>
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 10px; font-weight: 700; color: #64748B; letter-spacing: 0.05em; text-transform: uppercase;">
            Executive Intelligence Hub
        </div>
    </div>
""", unsafe_allow_html=True)
st.sidebar.subheader("Alert Settings")

st.sidebar.subheader("Advisory Reasoning Engine")
use_llm_agent = st.sidebar.toggle(
    "👔 Enable Executive Advisory Insights",
    value=False,
    help="Enable the executive reasoning client to formulate structured multi-step employee retention strategies based on corporate policies."
)

# Interactive email overrides
st.sidebar.subheader("SMTP Server Settings")
smtp_server = st.sidebar.text_input("SMTP Server", value=notification_module.SMTP_SERVER if notification_module.SMTP_SERVER else "smtp.gmail.com")
smtp_port = st.sidebar.number_input("SMTP Port", value=int(notification_module.SMTP_PORT) if notification_module.SMTP_PORT else 587, step=1)
smtp_user = st.sidebar.text_input("SMTP Username (Email)", value=notification_module.SMTP_USERNAME if notification_module.SMTP_USERNAME else "")
smtp_pass = st.sidebar.text_input("SMTP Password (App Password)", value=notification_module.SMTP_PASSWORD if notification_module.SMTP_PASSWORD else "", type="password")

st.sidebar.subheader("Email Headers")
sender = st.sidebar.text_input("Sender Email Address", value=sender_default if 'sender_default' in locals() else (SENDER_EMAIL if SENDER_EMAIL else "churn-agent@company.com"))
receiver = st.sidebar.text_input("HR Recipient Email Address", value=hr_default if 'hr_default' in locals() else (HR_EMAIL if HR_EMAIL else "hr@company.com"))

st.sidebar.info("Enter your Gmail address and 16-character App Password above. Leave SMTP Password empty to safely run in local HTML simulation mode.")

# =============================================================
# TITLE HEADER
# =============================================================
st.markdown('<h1 style="text-align: center;">Employee Churn & Retention Intelligence Hub</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #64748B; font-weight: 500; font-size: 15px; margin-top: -10px; margin-bottom: 20px;">Automated risk assessment, SHAP explainability analysis, and smart retention actions.</p>', unsafe_allow_html=True)
st.markdown("---")

# =============================================================
# TAB LAYOUTS
# =============================================================
tab1, tab2, tab3 = st.tabs(["📊 Organization Analytics Overview", "👤 Individual Risk Diagnostics & Intervention", "🔬 Strategic What-If Sandbox Simulation"])

# -------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------
with tab1:
    # Top Filter Row (mocking the Reports inline selectors in the mockup)
    st.markdown('<div class="card-header" style="margin-top: 5px; margin-bottom: 8px;">Workforce Filter Presets</div>', unsafe_allow_html=True)
    fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 2.5, 2, 2.5])
    fcol1.selectbox("Timeframe", options=["All-time", "Current Quarter", "Last Quarter", "Year-to-Date"], key="top_timeframe")
    risk_filter = fcol2.multiselect("Risk Tier", options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], default=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], key="top_risk")
    match_filter = fcol3.selectbox("Accuracy Filter", options=['All Records', 'Match ✓ Only', 'Error ✗ Only'], key="top_accuracy")
    search_id = fcol4.text_input("Search Employee ID", placeholder="Enter ID (e.g. 1024)", key="top_search")
    
    # Compile dynamic filtered dataframe based on selectors
    filtered_df = results_df.copy()
    if risk_filter:
        filtered_df = filtered_df[filtered_df['Risk Level'].isin(risk_filter)]
    if match_filter == 'Match ✓ Only':
        filtered_df = filtered_df[filtered_df['Prediction Match'] == 'Match ✓']
    elif match_filter == 'Error ✗ Only':
        filtered_df = filtered_df[filtered_df['Prediction Match'] == 'Error ✗']
    if search_id:
        try:
            filtered_df = filtered_df[filtered_df['Employee ID'] == int(search_id)]
        except ValueError:
            st.warning("Please type a valid integer Employee ID.")

    st.write("")
    
    # 1. Key Statistics Cards with SVG Sparkline trends
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_assessed = len(filtered_df)
    critical_cnt = (filtered_df['Risk Level'] == 'CRITICAL').sum()
    high_cnt = (filtered_df['Risk Level'] == 'HIGH').sum()
    med_cnt = (filtered_df['Risk Level'] == 'MEDIUM').sum()
    overall_acc = (filtered_df['Prediction Match'] == 'Match ✓').mean() * 100 if len(filtered_df) > 0 else 100.0
    
    spark_blue = """
    <svg class="sparkline-svg" viewBox="0 0 100 25" preserveAspectRatio="none" style="overflow: visible;">
        <defs>
            <linearGradient id="grad-blue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#2563EB" stop-opacity="0.15"/>
                <stop offset="100%" stop-color="#2563EB" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <path d="M0,15 C15,5 30,20 45,10 C60,2 75,18 90,8 L100,12 L100,25 L0,25 Z" fill="url(#grad-blue)"/>
        <path d="M0,15 C15,5 30,20 45,10 C60,2 75,18 90,8 L100,12" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
    
    spark_red = """
    <svg class="sparkline-svg" viewBox="0 0 100 25" preserveAspectRatio="none" style="overflow: visible;">
        <defs>
            <linearGradient id="grad-red" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#EF4444" stop-opacity="0.15"/>
                <stop offset="100%" stop-color="#EF4444" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <path d="M0,18 C15,10 30,22 45,8 C60,25 75,5 90,18 L100,15 L100,25 L0,25 Z" fill="url(#grad-red)"/>
        <path d="M0,18 C15,10 30,22 45,8 C60,25 75,5 90,18 L100,15" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
    
    spark_orange = """
    <svg class="sparkline-svg" viewBox="0 0 100 25" preserveAspectRatio="none" style="overflow: visible;">
        <defs>
            <linearGradient id="grad-orange" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#F97316" stop-opacity="0.15"/>
                <stop offset="100%" stop-color="#F97316" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <path d="M0,10 C15,15 30,5 45,18 C60,8 75,20 90,12 L100,10 L100,25 L0,25 Z" fill="url(#grad-orange)"/>
        <path d="M0,10 C15,15 30,5 45,18 C60,8 75,20 90,12 L100,10" fill="none" stroke="#F97316" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
    
    spark_yellow = """
    <svg class="sparkline-svg" viewBox="0 0 100 25" preserveAspectRatio="none" style="overflow: visible;">
        <defs>
            <linearGradient id="grad-yellow" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#FBBF24" stop-opacity="0.15"/>
                <stop offset="100%" stop-color="#FBBF24" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <path d="M0,12 C15,8 30,15 45,10 C60,14 75,8 90,12 L100,10 L100,25 L0,25 Z" fill="url(#grad-yellow)"/>
        <path d="M0,12 C15,8 30,15 45,10 C60,14 75,8 90,12 L100,10" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
    
    spark_green = """
    <svg class="sparkline-svg" viewBox="0 0 100 25" preserveAspectRatio="none" style="overflow: visible;">
        <defs>
            <linearGradient id="grad-green" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#10B981" stop-opacity="0.15"/>
                <stop offset="100%" stop-color="#10B981" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <path d="M0,20 C15,18 30,15 45,12 C60,8 75,6 90,4 L100,2 L100,25 L0,25 Z" fill="url(#grad-green)"/>
        <path d="M0,20 C15,18 30,15 45,12 C60,8 75,6 90,4 L100,2" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
    
    col1.markdown(f'<div class="metric-card"><div class="metric-label">Total Assessed</div><div class="metric-value" style="color: #2563EB;">{total_assessed}</div>{spark_blue}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><div class="metric-label">Critical Risks</div><div class="metric-value" style="color: #EF4444;">{critical_cnt}</div>{spark_red}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><div class="metric-label">High Risks</div><div class="metric-value" style="color: #F97316;">{high_cnt}</div>{spark_orange}</div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><div class="metric-label">Medium Risks</div><div class="metric-value" style="color: #FBBF24;">{med_cnt}</div>{spark_yellow}</div>', unsafe_allow_html=True)
    col5.markdown(f'<div class="metric-card"><div class="metric-label">Overall Accuracy</div><div class="metric-value" style="color: #10B981;">{overall_acc:.1f}%</div>{spark_green}</div>', unsafe_allow_html=True)
    
    st.write("")
    
    # 1.5 Workforce Attrition Drivers & Retention Anchors Cards (matching Weakest/Strongest Topics)
    dcol1, dcol2 = st.columns(2)
    
    with dcol1:
        st.markdown('<div class="card-header">Top Attrition Risk Drivers</div>', unsafe_allow_html=True)
        st.markdown(textwrap.dedent("""
            <div class="driver-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="driver-icon-container" style="background-color: #FFF1F2; color: #E11D48;">⏱️</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 700; color: #0F172A;">OverTime Frequency</div>
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">High overtime load triggers active exit search</div>
                        </div>
                    </div>
                    <div style="font-size: 13px; font-weight: 700; color: #E11D48;">87% Impact</div>
                </div>
                <div class="driver-progress-bg">
                    <div class="driver-progress-bar" style="width: 87%; background: linear-gradient(90deg, #F43F5E 0%, #FB7185 100%);"></div>
                </div>
            </div>
            
            <div class="driver-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="driver-icon-container" style="background-color: #FFF1F2; color: #E11D48;">🧠</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 700; color: #0F172A;">Psychological Exhaustion</div>
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">Workload and stress leading to high burnout</div>
                        </div>
                    </div>
                    <div style="font-size: 13px; font-weight: 700; color: #E11D48;">76% Impact</div>
                </div>
                <div class="driver-progress-bg">
                    <div class="driver-progress-bar" style="width: 76%; background: linear-gradient(90deg, #F43F5E 0%, #FB7185 100%);"></div>
                </div>
            </div>
            
            <div class="driver-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="driver-icon-container" style="background-color: #FFF5F1; color: #F97316;">🚗</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 700; color: #0F172A;">Long Work Commute</div>
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">Long distance transit amplifies fatigue levels</div>
                        </div>
                    </div>
                    <div style="font-size: 13px; font-weight: 700; color: #F97316;">63% Impact</div>
                </div>
                <div class="driver-progress-bg">
                    <div class="driver-progress-bar" style="width: 63%; background: linear-gradient(90deg, #F97316 0%, #FDBA74 100%);"></div>
                </div>
            </div>
        """).replace("\n", " "), unsafe_allow_html=True)
        
    with dcol2:
        st.markdown('<div class="card-header">Top Retention Stabilizers</div>', unsafe_allow_html=True)
        st.markdown(textwrap.dedent("""
            <div class="driver-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="driver-icon-container" style="background-color: #ECFDF5; color: #059669;">🏥</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 700; color: #0F172A;">Medical Insurance Coverage</div>
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">Comprehensive family coverage keeps loyalty high</div>
                        </div>
                    </div>
                    <div style="font-size: 13px; font-weight: 700; color: #059669;">95% Strength</div>
                </div>
                <div class="driver-progress-bg">
                    <div class="driver-progress-bar" style="width: 95%; background: linear-gradient(90deg, #10B981 0%, #34D399 100%);"></div>
                </div>
            </div>
            
            <div class="driver-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="driver-icon-container" style="background-color: #ECFDF5; color: #059669;">🏆</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 700; color: #0F172A;">Public Performance Recognition</div>
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">Consistent appreciation programs boost motivation</div>
                        </div>
                    </div>
                    <div style="font-size: 13px; font-weight: 700; color: #059669;">89% Strength</div>
                </div>
                <div class="driver-progress-bg">
                    <div class="driver-progress-bar" style="width: 89%; background: linear-gradient(90deg, #10B981 0%, #34D399 100%);"></div>
                </div>
            </div>
            
            <div class="driver-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="driver-icon-container" style="background-color: #ECFDF5; color: #059669;">😊</div>
                        <div>
                            <div style="font-size: 14px; font-weight: 700; color: #0F172A;">Job Role & Task Satisfaction</div>
                            <div style="font-size: 11px; color: #64748B; font-weight: 500;">Aligning tasks to skills reduces voluntary friction</div>
                        </div>
                    </div>
                    <div style="font-size: 13px; font-weight: 700; color: #059669;">84% Strength</div>
                </div>
                <div class="driver-progress-bg">
                    <div class="driver-progress-bar" style="width: 84%; background: linear-gradient(90deg, #10B981 0%, #34D399 100%);"></div>
                </div>
            </div>
        """).replace("\n", " "), unsafe_allow_html=True)
        
    st.write("")
    
    # 2. Visualization Row
    gcol1, gcol2 = st.columns([3, 2])
    
    with gcol1:
        st.markdown('<div class="card-header">Churn Risk Distribution</div>', unsafe_allow_html=True)
        # Custom matplotlib bar chart for distributions
        counts = [
            (results_df['Risk Level'] == 'CRITICAL').sum(),
            (results_df['Risk Level'] == 'HIGH').sum(),
            (results_df['Risk Level'] == 'MEDIUM').sum(),
            (results_df['Risk Level'] == 'LOW').sum()
        ]
        labels = ['🔴 CRITICAL (≥80%)', '🟠 HIGH (≥60%)', '🟡 MEDIUM (≥40%)', '🟢 LOW (<40%)']
        colors = ['#D32F2F', '#F57C00', '#FBC02D', '#388E3C']
        
        fig, ax = plt.subplots(figsize=(8, 3.8))
        bars = ax.bar(labels, counts, color=colors, edgecolor='none', width=0.55)
        ax.set_ylabel('Number of Employees', fontsize=10, fontweight='bold', color='#555555')
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=8)
        sns.despine()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add labels to bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    with gcol2:
        st.markdown('<div class="card-header">Model Metrics by Risk Level</div>', unsafe_allow_html=True)
        acc_by_level = []
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            subset = results_df[results_df['Risk Level'] == level]
            correct = (subset['Prediction Match'] == 'Match ✓').sum()
            total = len(subset)
            acc = (correct / total * 100) if total > 0 else 0.0
            acc_by_level.append({'Risk Level': level, 'Accuracy': f'{acc:.1f}%', 'Correct': f'{correct}/{total}'})
        
        acc_df = pd.DataFrame(acc_by_level)
        st.dataframe(acc_df, use_container_width=True, hide_index=True)
        st.markdown("""
            * **Critical/High Accuracy** indicates that the model is extremely precise when assigning high churn probability scores.
            * Retention efforts should be prioritized starting from the **Critical** tier.
        """)
        
    st.write("")
    
    # 2.5 High-Fidelity Workforce Leaderboards (matching User & Groups Leaderboards in mockup)
    lcol1, lcol2 = st.columns(2)
        
    with lcol1:
        st.markdown('<div class="card-header">Highest Attrition Risk Profiles</div>', unsafe_allow_html=True)
        top_risk_emps = results_df.sort_values(by='Churn Probability', ascending=False).head(3)
        
        html_rows = ""
        rank = 1
        colors = ["#EF4444", "#F97316", "#F59E0B"]
        for _, row in top_risk_emps.iterrows():
            emp_id = int(row['Employee ID'])
            name = get_employee_name(emp_id)
            prob = row['Churn Probability']
            initials = "".join([part[0] for part in name.split() if part])[:2]
            
            html_rows += textwrap.dedent(f"""
            <div class="leaderboard-row">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div class="avatar-circle" style="background-color: {colors[rank-1]};">{initials}</div>
                    <div>
                        <div style="font-size: 14px; font-weight: 700; color: #0F172A;">{name}</div>
                        <div style="font-size: 11px; color: #64748B; font-weight: 500;">Employee ID: #{emp_id} | Churn Probability</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="font-size: 15px; font-weight: 800; color: #EF4444;">{prob:.1f}%</div>
                    <div class="rank-badge" style="color: #EF4444;">
                        <span>{rank}</span><span style="font-size: 10px;">▲</span>
                    </div>
                </div>
            </div>
            """)
            rank += 1
            
        st.markdown(f'<div class="leaderboard-card">{html_rows}</div>', unsafe_allow_html=True)
        
    with lcol2:
        st.markdown('<div class="card-header">Stable Workforce Profiles</div>', unsafe_allow_html=True)
        stable_emps = results_df.sort_values(by='Churn Probability', ascending=True).head(3)
        
        html_rows = ""
        rank = 1
        colors = ["#10B981", "#059669", "#34D399"]
        for _, row in stable_emps.iterrows():
            emp_id = int(row['Employee ID'])
            name = get_employee_name(emp_id)
            prob = row['Churn Probability']
            initials = "".join([part[0] for part in name.split() if part])[:2]
            
            html_rows += textwrap.dedent(f"""
            <div class="leaderboard-row">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div class="avatar-circle" style="background-color: {colors[rank-1]};">{initials}</div>
                    <div>
                        <div style="font-size: 14px; font-weight: 700; color: #0F172A;">{name}</div>
                        <div style="font-size: 11px; color: #64748B; font-weight: 500;">Employee ID: #{emp_id} | Churn Probability</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="font-size: 15px; font-weight: 800; color: #10B981;">{prob:.1f}%</div>
                    <div class="rank-badge" style="color: #10B981;">
                        <span>{rank}</span><span style="font-size: 10px;">▼</span>
                    </div>
                </div>
            </div>
            """)
            rank += 1
            
        st.markdown(f'<div class="leaderboard-card">{html_rows}</div>', unsafe_allow_html=True)
        
    st.write("")
    
    # 3. Filterable Employee Risk Database Table (streamlined to use global filters from page top)
    st.markdown('<div class="card-header">Filterable Employee Risk Database</div>', unsafe_allow_html=True)
    
    st.dataframe(
        filtered_df.drop(columns=['Risk Level']),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn Probability": st.column_config.ProgressColumn(
                "Churn Probability (%)",
                help="Prediction probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            )
        }
    )

    # 4. Interactive Pearson Correlation Explorer
    st.write("")
    with st.expander("📊 Interactive Pearson Correlation Matrix & Feature Explorer", expanded=False):
        st.markdown("Analyze linear relationships between key organizational metrics. A Pearson correlation coefficient ($r$) close to +1 indicates a strong positive correlation, while -1 indicates a strong negative correlation.")
        
        # Combine training features and labels to calculate Pearson coefficients
        correlation_data = X_train.copy()
        correlation_data['Attrition'] = y_train
        
        # Friendly feature names mapping for multi-select
        default_selected_features = [
            'Attrition',
            'Job_Satisfaction',
            'MonthlySalary',
            'OverTime',
            'Work_Live_Balance',
            'Psychological_Exhaustion',
            'Get_Deserved_Promotion',
            'Environment_Satisfaction',
            'Years_Experience'
        ]
        
        selected_corr_features = st.multiselect(
            "Select Features for Correlation Analysis",
            options=list(correlation_data.columns),
            default=default_selected_features,
            help="Choose which organizational attributes to display in the Pearson correlation matrix."
        )
        
        if len(selected_corr_features) > 1:
            # Calculate Pearson correlation matrix dynamically
            corr_matrix = correlation_data[selected_corr_features].corr(method='pearson')
            
            # Draw premium Seaborn heatmap
            fig, ax = plt.subplots(figsize=(10, 7.5))
            
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                linewidths=0.5,
                annot_kws={'size': 9, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8},
                ax=ax
            )
            
            ax.set_title("Pearson Correlation Heatmap ($r$)", fontsize=12, fontweight='bold', pad=15)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(fontsize=9)
            plt.tight_layout()
            
            st.pyplot(fig)
        else:
            st.warning("Please select at least 2 features to generate the Pearson correlation matrix.")

# -------------------------------------------------------------
# TAB 2: INDIVIDUAL RISK DIAGNOSTICS
# -------------------------------------------------------------
with tab2:
    if 'sent_emails' not in st.session_state:
        st.session_state['sent_emails'] = set()
        
    icol1, icol2 = st.columns([1, 2])
    
    with icol1:
        st.markdown('<div class="card-header">Employee Selector</div>', unsafe_allow_html=True)
        # Construct sorted dropdown highlighting highest risk employees first
        dropdown_options = []
        sorted_results = results_df.sort_values(by='Churn Probability', ascending=False)
        for _, row in sorted_results.iterrows():
            emp_id = int(row['Employee ID'])
            name = get_employee_name(emp_id)
            dropdown_options.append(f"{name} (ID: #{emp_id} | Prob: {row['Churn Probability']:.1f}% | {row['Status']})")
            
        selected_option = st.selectbox("Choose Employee Profile", options=dropdown_options)
        
        # Parse selected employee ID
        import re
        match_id = re.search(r'ID: #(\d+)', selected_option)
        if match_id:
            selected_emp_id = int(match_id.group(1))
        else:
            selected_emp_id = int(selected_option.split(' ')[0].replace('#', ''))
            
        emp_idx = X_test.index.get_loc(selected_emp_id)
        
        # Calculate single stats
        emp_prob = probabilities[emp_idx]
        emp_level, emp_emoji, emp_color = classify_risk(emp_prob)
        emp_actual = y_test.iloc[emp_idx]
        emp_actual_str = "LEFT ✗" if emp_actual == 1 else "STAYED ✓"
        emp_actual_color = "#D32F2F" if emp_actual == 1 else "#388E3C"
        
        # Display high-fidelity profile card matching mockup circular avatar style
        emp_name = get_employee_name(selected_emp_id)
        emp_initials = "".join([part[0] for part in emp_name.split() if part])[:2]
        
        st.write("")
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 24px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px; border-bottom: 1px solid #F1F5F9; padding-bottom: 16px;">
                    <div class="avatar-circle" style="background-color: {emp_color}; width: 48px; height: 48px; font-size: 16px; font-weight: bold; color: white; display: flex; align-items: center; justify-content: center; border-radius: 50%;">
                        {emp_initials}
                    </div>
                    <div>
                        <div style="font-size: 18px; font-weight: 800; color: #0F172A;">{emp_name}</div>
                        <div style="font-size: 12px; color: #64748B; font-weight: 500;">Employee ID: <strong>#{selected_emp_id}</strong></div>
                    </div>
                </div>
                
                <div style="text-align: center; background-color: {emp_color}0C; border: 1px solid {emp_color}40; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                    <div style="font-size: 11px; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Churn Probability</div>
                    <div style="font-size: 38px; font-weight: 800; color: {emp_color}; margin: 6px 0; letter-spacing: -0.02em;">{emp_prob:.1%}</div>
                    <span style="background-color: {emp_color}; color: white; padding: 4px 12px; font-size: 11px; font-weight: 800; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;">
                        {emp_emoji} {emp_level} RISK LEVEL
                    </span>
                </div>
                
                <div style="display: flex; align-items: center; justify-content: space-between; font-size: 13px; color: #334155; padding: 10px 14px; background-color: #F8FAFC; border-radius: 8px;">
                    <span style="font-weight: 500;">Ground Truth Status:</span>
                    <strong style="color: {emp_actual_color};">{emp_actual_str}</strong>
                </div>
            </div>
        """).replace("\n", " "), unsafe_allow_html=True)
        
        # Email Trigger Form
        st.markdown('<div class="card-header">Retention Alert Transmitter</div>', unsafe_allow_html=True)
        st.markdown("Deliver the generated retention profile and recommendations directly to HR's inbox.")
        
        # Assemble feature recommendations for email delivery and display
        emp_series = X_test.iloc[emp_idx]
        emp_shap = sv[emp_idx]
        
        if use_llm_agent:
            @st.cache_data
            def get_cached_agentic_recommendations(emp_id, _series, _shap):
                return get_agentic_recommendations_level3(_series, _shap)
            
            with st.spinner("💼 Formulating strategic retention insights and verifying indicators..."):
                try:
                    agent_res = get_cached_agentic_recommendations(selected_emp_id, emp_series, emp_shap)
                    thought_process = agent_res.get("thought_process", "Reasoning completed.")
                    recs = agent_res.get("recommendations", [])
                    
                    # AUTOMATED EMAIL TRANSMISSION FOR LEVEL 3 ADVISORY
                    if selected_emp_id not in st.session_state['sent_emails']:
                        # Overwrite SMTP settings dynamically with user inputs
                        notification_module.SMTP_SERVER = smtp_server
                        notification_module.SMTP_PORT = int(smtp_port)
                        notification_module.SMTP_USERNAME = smtp_user
                        notification_module.SMTP_PASSWORD = smtp_pass
                        
                        success, msg = send_retention_email(
                            employee_id=selected_emp_id,
                            risk_level=emp_level,
                            probability=emp_prob,
                            recommendations=recs,
                            actual_label=int(emp_actual),
                            custom_sender=sender,
                            custom_receiver=receiver
                        )
                        if success:
                            st.session_state['sent_emails'].add(selected_emp_id)
                            st.toast(f"📧 Advisory email dispatched automatically for Employee #{selected_emp_id}!", icon="✉️")
                        else:
                            st.error(f"Failed to transmit automated email: {msg}")
                except Exception as e:
                    st.error(f"Error calling Executive Advisory Engine: {str(e)}")
                    recs = get_recommendations(emp_series, emp_shap)
                    thought_process = "Error occurred. Falling back to rule-based recommendations."
        else:
            recs = get_recommendations(emp_series, emp_shap)
            thought_process = None
        
        # Render manual email trigger button
        if use_llm_agent:
            st.button("📧 Churn Alert Email Sent Automatically", use_container_width=True, disabled=True)
        else:
            if st.button("📧 Send Churn Alert Email", use_container_width=True):
                with st.spinner("Processing email delivery..."):
                    # Overwrite module-level SMTP configuration dynamically with user sidebar inputs
                    notification_module.SMTP_SERVER = smtp_server
                    notification_module.SMTP_PORT = int(smtp_port)
                    notification_module.SMTP_USERNAME = smtp_user
                    notification_module.SMTP_PASSWORD = smtp_pass
                    
                    success, msg = send_retention_email(
                        employee_id=selected_emp_id,
                        risk_level=emp_level,
                        probability=emp_prob,
                        recommendations=recs,
                        actual_label=int(emp_actual),
                        custom_sender=sender,
                        custom_receiver=receiver
                    )
                    
                if success:
                    st.success("Alert sent successfully!")
                    st.toast(msg, icon="✅")
                else:
                    st.error(f"Failed to transmit email alert.")
                    st.info(msg)
                    st.toast("Email delivery error", icon="❌")
                
    with icol2:
        st.markdown('<div class="card-header">Individual Diagnostics & Explainable AI (SHAP)</div>', unsafe_allow_html=True)
        
        # Display tabular feature values of the employee
        st.write("**Active Employee Feature Values:**")
        
        emp_features = X_test.iloc[emp_idx]
        features_data = []
        for col, val in emp_features.items():
            # Only display key visual features to keep it neat
            if col in FEATURE_LABELS:
                readable_val = FEATURE_LABELS[col].get(int(round(val)), str(val))
                features_data.append({'Feature Name': col.replace('_', ' '), 'Raw Score': val, 'HR Setting': readable_val})
                
        feat_df_display = pd.DataFrame(features_data).sort_values(by='Feature Name')
        with st.expander("🔍 View Complete Feature Profile Grid", expanded=False):
            st.dataframe(feat_df_display, use_container_width=True, hide_index=True)
            
        # Draw dynamic SHAP plots
        st.write("")
        st.write("**Top Risk Drivers (Why does model predict Churn?)**")
        
        # Build individual SHAP drivers
        shap_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Value': emp_features.values,
            'SHAP': emp_shap
        })
        
        # Separate positive impact (pushing to leave) and negative impact (pushing to stay)
        shap_df['Feature_Label'] = shap_df.apply(
            lambda r: f"{r['Feature'].replace('_', ' ')} = {FEATURE_LABELS.get(r['Feature'], {}).get(int(round(r['Value'])), str(r['Value']))}", axis=1
        )
        
        # Sort and take top 8 features influencing the prediction the most
        top_shap_plot = shap_df.reindex(shap_df['SHAP'].abs().sort_values(ascending=False).index).head(8)
        top_shap_plot = top_shap_plot.sort_values(by='SHAP', ascending=True) # Ascending for horizontal plotting
        
        # Plot
        fig, ax = plt.subplots(figsize=(9, 4))
        bar_colors = ['#D32F2F' if x > 0 else '#388E3C' for x in top_shap_plot['SHAP']]
        
        bars = ax.barh(top_shap_plot['Feature_Label'], top_shap_plot['SHAP'], color=bar_colors, edgecolor='none', height=0.6)
        ax.axvline(x=0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('SHAP Impact on Churn (Red = Attrition, Green = Retention)', fontsize=9, fontweight='bold', color='#555555')
        ax.tick_params(axis='y', labelsize=8.5)
        ax.tick_params(axis='x', labelsize=8)
        sns.despine()
        plt.grid(axis='x', linestyle='--', alpha=0.2)
        
        # Expand x-limits to avoid y-label overlap
        xmin, xmax = ax.get_xlim()
        x_margin = (xmax - xmin) * 0.22
        ax.set_xlim(xmin - x_margin, xmax + x_margin)
        
        # Draw labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.05), bar.get_y() + bar.get_height()/2.,
                    f'{width:+.3f}', ha='left' if width >= 0 else 'right', va='center',
                    fontsize=8, fontweight='bold', color='#333333')
                    
        plt.tight_layout()
        st.pyplot(fig)
        
        # Actionable recommendations
        if use_llm_agent and 'thought_process' in locals() and thought_process:
            st.write("")
            st.markdown('<div class="card-header">📋 Strategic Root-Cause Diagnostics & Executive Rationale</div>', unsafe_allow_html=True)
            st.info(thought_process)
            
        st.write("")
        st.markdown('<div class="card-header">Tailored Actionable HR Recommendations</div>', unsafe_allow_html=True)
        
        def get_recommendation_theme(feature: str) -> tuple:
            f = feature.lower()
            if 'satisfaction' in f or 'engagement' in f:
                return ("😊", "#ECFDF5", "#059669", "#10B981")
            elif 'exhaustion' in f or 'stress' in f or 'health' in f:
                return ("🧠", "#FFF1F2", "#E11D48", "#F43F5E")
            elif 'balance' in f or 'travel' in f or 'commute' in f or 'distance' in f:
                return ("🏡", "#EFF6FF", "#2563EB", "#3B82F6")
            elif 'salary' in f or 'promotion' in f or 'bonus' in f or 'rewards' in f:
                return ("💰", "#FEF3C7", "#D97706", "#F59E0B")
            elif 'overtime' in f:
                return ("⏱️", "#FFF5F1", "#F97316", "#F97316")
            elif 'insurance' in f:
                return ("🏥", "#ECFDF5", "#059669", "#10B981")
            else:
                return ("📋", "#F1F5F9", "#475569", "#64748B")

        if recs:
            for idx_rec, r in enumerate(recs, 1):
                badge_bg = "#EF4444" if r['priority'] == 'URGENT' else "#3B82F6"
                badge_color = "#FFFFFF"
                
                # Fetch theme details
                rec_icon, rec_bg, rec_text_color, rec_accent = get_recommendation_theme(r['feature'])
                
                # Mini SHAP progress bar percentage
                shap_pct = min(100, int((r['shap_score'] / 0.35) * 100)) if r['shap_score'] > 0 else 5
                
                st.markdown(textwrap.dedent(f"""
                    <div class="intervention-card">
                        <div class="intervention-row">
                            <div class="intervention-left">
                                <div class="intervention-badge-icon" style="background-color: {rec_bg}; color: {rec_text_color};">
                                    {rec_icon}
                                </div>
                                <div>
                                    <div style="font-size: 15px; font-weight: 700; color: #0F172A;">{idx_rec}. {r['feature'].replace('_', ' ')}</div>
                                    <div style="font-size: 11px; color: #64748B; font-weight: 500; margin-top: 2px;">
                                        Current Status: <strong style="color: #334155;">{r['value_label']}</strong> | SHAP Impact: <strong style="color: {rec_text_color};">{r['shap_score']:+.4f}</strong>
                                    </div>
                                </div>
                            </div>
                            <div class="intervention-right">
                                <span class="priority-badge" style="background-color: {badge_bg}; color: {badge_color};">{r['priority']}</span>
                            </div>
                        </div>
                        
                        <div class="driver-progress-bg" style="margin-top: 10px; margin-bottom: 12px; height: 5px;">
                            <div class="driver-progress-bar" style="width: {shap_pct}%; background: {rec_accent};"></div>
                        </div>
                        
                        <div style="padding: 12px 14px; background-color: #F8FAFC; border-radius: 8px; border-left: 3px solid {rec_accent}; font-size: 13px; color: #334155; line-height: 1.45;">
                            <strong>Strategic Action Plan:</strong> {r['action']}
                        </div>
                    </div>
                """).replace("\n", " "), unsafe_allow_html=True)
        else:
            st.info("No customized risk actions found for this feature configuration. Ensure regular touchpoints and baseline wellness checkins.")


# -------------------------------------------------------------
# TAB 3: STRATEGIC WHAT-IF SANDBOX SIMULATION
# -------------------------------------------------------------
with tab3:
    st.markdown('<div class="card-header">🔬 Workforce Risk Simulator & What-If Projection</div>', unsafe_allow_html=True)
    st.markdown("Assess hypothetical employee profiles dynamically or test adjustments to existing profiles to see how risk probabilities shift in real-time.")
    st.write("")
    
    # Selection of base template
    template_option = st.selectbox("Pre-populate Simulator with Base Profile", options=["Workforce Average Profile"] + [f"Employee #{idx}" for idx in X_test.index])
    
    # Initialize base data
    base_profile = pd.Series(0.0, index=X_test.columns)
    if template_option == "Workforce Average Profile":
        # Compute median/mode values
        for col in X_test.columns:
            if col in FEATURE_LABELS:
                base_profile[col] = float(X_test[col].mode().iloc[0])
            else:
                base_profile[col] = float(X_test[col].median())
    else:
        # Load specific employee values
        selected_template_id = int(template_option.split('#')[1])
        base_profile = X_test.loc[selected_template_id].copy()
        
    st.write("---")
    st.subheader("Adjust Employee Attributes")
    
    col_a, col_b, col_c = st.columns(3)
    
    sim_values = base_profile.copy()
    
    # Render selectors based on FEATURE_LABELS
    with col_a:
        st.markdown("**Job & Compensation**")
        for feat in ['MonthlySalary', 'OverTime', 'Payment_Overtime', 'Get_Deserved_Promotion', 'Bonus', 'Job_Opportunities', 'Job_Stability']:
            if feat in FEATURE_LABELS:
                options_dict = FEATURE_LABELS[feat]
                current_val = int(round(sim_values[feat]))
                # Guard against out of bounds
                if current_val not in options_dict:
                    current_val = list(options_dict.keys())[0]
                
                # Format options list for selectbox
                labels_list = [f"{k}: {v}" for k, v in options_dict.items()]
                default_index = list(options_dict.keys()).index(current_val)
                
                sel_label = st.selectbox(feat.replace('_', ' '), options=labels_list, index=default_index, key=f"sim_{feat}")
                sim_values[feat] = float(int(sel_label.split(':')[0]))
                
    with col_b:
        st.markdown("**Well-being & Environment**")
        for feat in ['Job_Satisfaction', 'Psychological_Exhaustion', 'Physical_Stress', 'Work_Live_Balance', 'Environment_Satisfaction', 'Job_Support', 'Job_Engagement']:
            if feat in FEATURE_LABELS:
                options_dict = FEATURE_LABELS[feat]
                current_val = int(round(sim_values[feat]))
                if current_val not in options_dict:
                    current_val = list(options_dict.keys())[0]
                    
                labels_list = [f"{k}: {v}" for k, v in options_dict.items()]
                default_index = list(options_dict.keys()).index(current_val)
                
                sel_label = st.selectbox(feat.replace('_', ' '), options=labels_list, index=default_index, key=f"sim_{feat}")
                sim_values[feat] = float(int(sel_label.split(':')[0]))
                
    with col_c:
        st.markdown("**Demographics & Profile**")
        for feat in ['Age', 'Academic_degree', 'Years_Experience', 'Years_experience_lastorganization', 'Business_Travel', 'Distance_to_work', 'Health_Issues']:
            if feat in FEATURE_LABELS:
                options_dict = FEATURE_LABELS[feat]
                current_val = int(round(sim_values[feat]))
                if current_val not in options_dict:
                    current_val = list(options_dict.keys())[0]
                    
                labels_list = [f"{k}: {v}" for k, v in options_dict.items()]
                default_index = list(options_dict.keys()).index(current_val)
                
                sel_label = st.selectbox(feat.replace('_', ' '), options=labels_list, index=default_index, key=f"sim_{feat}")
                sim_values[feat] = float(int(sel_label.split(':')[0]))
                
    st.write("")
    run_sim = st.button("🔮 Calculate Retention Simulation", use_container_width=True)
    
    if run_sim:
        # Wrap in a single-row DataFrame
        sim_df = pd.DataFrame([sim_values], columns=X_test.columns)
        
        # Predict probability
        sim_prob = model.predict_proba(sim_df)[0, 1]
        sim_level, sim_emoji, sim_color = classify_risk(sim_prob)
        
        st.write("---")
        st.markdown('<div class="card-header">🔮 Simulation Diagnostics & Projection Outcomes</div>', unsafe_allow_html=True)
        
        # Display score
        st.markdown(f"""
            <div style="background-color:{sim_color}22; border: 2px solid {sim_color}; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px; max-width: 400px; margin: 0 auto;">
                <div style="font-size: 14px; color: #64748B; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em;">Projected Attrition Risk</div>
                <div style="font-size: 48px; font-weight: 800; color: {sim_color}; margin: 10px 0; letter-spacing: -0.02em;">{sim_prob:.1%}</div>
                <span style="background-color: {sim_color}; color: white; padding: 6px 16px; font-size: 14px; font-weight: bold; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.05em;">{sim_emoji} {sim_level} RISK LEVEL</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Compute dynamic SHAP values on-the-fly for this custom row
        with st.spinner("Calculating explainability models on custom projection..."):
            sim_explainer = shap.TreeExplainer(model)
            sim_shap_val = sim_explainer.shap_values(sim_df)
            if isinstance(sim_shap_val, list):
                row_sv = sim_shap_val[1][0]
            else:
                row_sv = sim_shap_val[0]
                
        # Draw dynamic SHAP horizontal bar chart
        st.write("**Top Attribute Drivers (Red = Promotes Attrition, Green = Promotes Retention)**")
        
        shap_sim_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Value': sim_values.values,
            'SHAP': row_sv
        })
        
        shap_sim_df['Feature_Label'] = shap_sim_df.apply(
            lambda r: f"{r['Feature'].replace('_', ' ')} = {FEATURE_LABELS.get(r['Feature'], {}).get(int(round(r['Value'])), str(r['Value']))}", axis=1
        )
        
        top_sim_plot = shap_sim_df.reindex(shap_sim_df['SHAP'].abs().sort_values(ascending=False).index).head(8)
        top_sim_plot = top_sim_plot.sort_values(by='SHAP', ascending=True)
        
        fig, ax = plt.subplots(figsize=(9, 4))
        bar_colors = ['#D32F2F' if x > 0 else '#388E3C' for x in top_sim_plot['SHAP']]
        
        bars = ax.barh(top_sim_plot['Feature_Label'], top_sim_plot['SHAP'], color=bar_colors, edgecolor='none', height=0.6)
        ax.axvline(x=0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('SHAP Impact on Churn (Red = Attrition, Green = Retention)', fontsize=9, fontweight='bold', color='#555555')
        ax.tick_params(axis='y', labelsize=8.5)
        ax.tick_params(axis='x', labelsize=8)
        sns.despine()
        plt.grid(axis='x', linestyle='--', alpha=0.2)
        
        # Expand x-limits to avoid y-label overlap
        xmin, xmax = ax.get_xlim()
        x_margin = (xmax - xmin) * 0.22
        ax.set_xlim(xmin - x_margin, xmax + x_margin)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.05), bar.get_y() + bar.get_height()/2.,
                    f'{width:+.3f}', ha='left' if width >= 0 else 'right', va='center',
                    fontsize=8, fontweight='bold', color='#333333')
                    
        plt.tight_layout()
        st.pyplot(fig)
        
        # Calculate dynamic recommendations (Level 3 or standard)
        st.write("")
        if use_llm_agent:
            with st.spinner("Formulating customized strategic insights from Advisory Engine..."):
                try:
                    sim_agent_res = get_agentic_recommendations_level3(sim_values, row_sv)
                    sim_thought = sim_agent_res.get("thought_process", "")
                    sim_recs = sim_agent_res.get("recommendations", [])
                except Exception as e:
                    st.error(f"Error calling Executive Advisory Engine: {str(e)}")
                    sim_recs = get_recommendations(sim_values, row_sv)
                    sim_thought = "Error occurred. Falling back to rule-based recommendations."
        else:
            sim_recs = get_recommendations(sim_values, row_sv)
            sim_thought = None
            
        if use_llm_agent and sim_thought:
            st.write("")
            st.markdown('<div class="card-header">📋 Strategic Root-Cause Diagnostics & Executive Rationale</div>', unsafe_allow_html=True)
            st.info(sim_thought)
            
        st.write("")
        st.markdown('<div class="card-header">Tailored Actionable HR Recommendations</div>', unsafe_allow_html=True)
        
        def get_recommendation_theme(feature: str) -> tuple:
            f = feature.lower()
            if 'satisfaction' in f or 'engagement' in f:
                return ("😊", "#ECFDF5", "#059669", "#10B981")
            elif 'exhaustion' in f or 'stress' in f or 'health' in f:
                return ("🧠", "#FFF1F2", "#E11D48", "#F43F5E")
            elif 'balance' in f or 'travel' in f or 'commute' in f or 'distance' in f:
                return ("🏡", "#EFF6FF", "#2563EB", "#3B82F6")
            elif 'salary' in f or 'promotion' in f or 'bonus' in f or 'rewards' in f:
                return ("💰", "#FEF3C7", "#D97706", "#F59E0B")
            elif 'overtime' in f:
                return ("⏱️", "#FFF5F1", "#F97316", "#F97316")
            elif 'insurance' in f:
                return ("🏥", "#ECFDF5", "#059669", "#10B981")
            else:
                return ("📋", "#F1F5F9", "#475569", "#64748B")

        if sim_recs:
            for idx_rec, r in enumerate(sim_recs, 1):
                badge_bg = "#EF4444" if r['priority'] == 'URGENT' else "#3B82F6"
                badge_color = "#FFFFFF"
                
                # Fetch theme details
                rec_icon, rec_bg, rec_text_color, rec_accent = get_recommendation_theme(r['feature'])
                
                # Mini SHAP progress bar percentage
                shap_pct = min(100, int((r['shap_score'] / 0.35) * 100)) if r['shap_score'] > 0 else 5
                
                st.markdown(textwrap.dedent(f"""
                    <div class="intervention-card">
                        <div class="intervention-row">
                            <div class="intervention-left">
                                <div class="intervention-badge-icon" style="background-color: {rec_bg}; color: {rec_text_color};">
                                    {rec_icon}
                                </div>
                                <div>
                                    <div style="font-size: 15px; font-weight: 700; color: #0F172A;">{idx_rec}. {r['feature'].replace('_', ' ')}</div>
                                    <div style="font-size: 11px; color: #64748B; font-weight: 500; margin-top: 2px;">
                                        Current Status: <strong style="color: #334155;">{r['value_label']}</strong> | SHAP Impact: <strong style="color: {rec_text_color};">{r['shap_score']:+.4f}</strong>
                                    </div>
                                </div>
                            </div>
                            <div class="intervention-right">
                                <span class="priority-badge" style="background-color: {badge_bg}; color: {badge_color};">{r['priority']}</span>
                            </div>
                        </div>
                        
                        <div class="driver-progress-bg" style="margin-top: 10px; margin-bottom: 12px; height: 5px;">
                            <div class="driver-progress-bar" style="width: {shap_pct}%; background: {rec_accent};"></div>
                        </div>
                        
                        <div style="padding: 12px 14px; background-color: #F8FAFC; border-radius: 8px; border-left: 3px solid {rec_accent}; font-size: 13px; color: #334155; line-height: 1.45;">
                            <strong>Strategic Action Plan:</strong> {r['action']}
                        </div>
                    </div>
                """).replace("\n", " "), unsafe_allow_html=True)
        else:
            st.info("No customized risk actions found for this simulated configuration.")
