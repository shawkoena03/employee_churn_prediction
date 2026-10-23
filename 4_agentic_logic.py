# =============================================================
# PHASE 4 — AGENT LOGIC
# Project : Employee Churn Prediction
# Dataset : Saudi Employee Attrition Dataset
# Agent   : Risk Assessment + Smart Recommendations + Email
# =============================================================
import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import pickle
import json
import os
import shap
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Fallback load env
def load_env(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    val = value.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ[key.strip()] = val

load_env()

import requests

# =============================================================
# CONFIGURATION
# =============================================================
OUTPUT_DIR   = "outputs"
REPORT_DIR   = os.path.join(OUTPUT_DIR, "agent_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Risk thresholds (churn probability)
RISK_THRESHOLDS = {
    'CRITICAL' : 0.80,   # >= 80%  → Critical
    'HIGH'     : 0.60,   # >= 60%  → High
    'MEDIUM'   : 0.40,   # >= 40%  → Medium
    'LOW'      : 0.00,   # <  40%  → Low
}

# HR email config (simulated — replace with real SMTP in production)
HR_EMAIL = "hr@company.com"
SENDER   = "churn-agent@company.com"

# =============================================================
# SECTION 1 — LOAD PHASE 3 OUTPUTS
# =============================================================
print("=" * 60)
print("SECTION 1 — LOADING PHASE 3 OUTPUTS")
print("=" * 60)

# Load tuned model
with open(os.path.join(OUTPUT_DIR, 'best_model_tuned.pkl'), 'rb') as f:
    model = pickle.load(f)

# Load test data
with open(os.path.join(OUTPUT_DIR, 'test_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_test  = data['X_test']
y_test  = data['y_test']
X_train = data['X_train']
y_train = data['y_train']

# Load SHAP feature importance ranking
shap_importance = pd.read_csv(os.path.join(OUTPUT_DIR, 'shap_feature_importance.csv'))
top_features    = shap_importance['Feature'].tolist()

print(f"✅ Model destroyed")
print(f"✅ Test set turning off       : {X_test.shape[0]} employees")
print(f"✅ Koena proof verification: {len(top_features)} features ranked")

# =============================================================
# SECTION 2 — COMPUTE SHAP VALUES FOR TEST SET
# =============================================================
print()
print("=" * 60)
print("SECTION 2 — COMPUTING SHAP VALUES")
print("           (This may take 1–2 minutes...)")
print("=" * 60)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle LightGBM SHAP output format
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

print(f"✅ SHAP values computed for {X_test.shape[0]} employees")

# =============================================================
# SECTION 3 — RISK CLASSIFICATION FUNCTION
# =============================================================

def classify_risk(probability: float) -> tuple:
    """
    Returns (risk_level, emoji, description) based on churn probability.
    4 levels: CRITICAL / HIGH / MEDIUM / LOW
    """
    if probability >= RISK_THRESHOLDS['CRITICAL']:
        return ('CRITICAL', '🔴', 'Immediate intervention required')
    elif probability >= RISK_THRESHOLDS['HIGH']:
        return ('HIGH',     '🟠', 'Urgent attention needed')
    elif probability >= RISK_THRESHOLDS['MEDIUM']:
        return ('MEDIUM',   '🟡', 'Monitor and follow up')
    else:
        return ('LOW',      '🟢', 'No immediate action required')


# =============================================================
# SECTION 4 — FEATURE-SPECIFIC RECOMMENDATION ENGINE
# =============================================================

# Maps each feature + its encoded value to a human-readable label
FEATURE_LABELS = {
    'Job_Satisfaction'                      : {0: 'Not Satisfied', 1: 'Satisfied', 2: 'Very Satisfied'},
    'Psychological_Exhaustion'              : {0: 'No', 1: 'Sometimes', 2: 'Yes'},
    'Physical_Stress'                       : {0: 'No', 1: 'Sometimes', 2: 'Yes'},
    'Work_Live_Balance'                     : {0: 'Easy', 1: 'Medium', 2: 'Difficult'},
    'MonthlySalary'                         : {0: '1k–5k SAR', 1: '6k–10k SAR', 2: '11k–15k SAR', 3: '16k+ SAR'},
    'Get_Deserved_Promotion'                : {0: 'No', 1: 'Yes'},
    'Age'                                   : {0: '21–30', 1: '31–40', 2: '41+'},
    'Business_Travel'                       : {0: 'No Travel', 1: 'Rarely', 2: 'Frequently'},
    'OverTime'                              : {0: 'No', 1: 'Yes'},
    'Payment_Overtime'                      : {0: 'No Overtime', 1: 'Overtime Not Paid', 2: 'Overtime Paid'},
    'Job_Opportunities'                     : {0: 'No', 1: 'Yes'},
    'Job_Stability'                         : {0: 'No', 1: 'Yes'},
    'Recognition'                           : {0: 'No', 1: 'Yes'},
    'Bonus'                                 : {0: 'No', 1: 'Yes'},
    'MedicalInsurance'                      : {0: 'No', 1: 'Yes'},
    'Rewards_Wages_Satisfaction'            : {0: 'No', 1: 'Yes'},
    'Useful_Training_Programs'              : {0: 'No', 1: 'Yes'},
    'Environment_Satisfaction'              : {0: 'Low', 1: 'Medium', 2: 'High'},
    'Emotional_Commitment'                  : {0: 'Low', 1: 'Medium', 2: 'High'},
    'Job_Engagement'                        : {0: 'Easy', 1: 'Medium', 2: 'Difficult'},
    'Job_Support'                           : {0: 'Low', 1: 'Medium', 2: 'High'},
    'Distance_to_work'                      : {0: 'Close', 1: 'Medium', 2: 'Far'},
    'Health_Issues'                         : {0: 'No', 1: 'Yes'},
    'Academic_degree'                       : {0: 'Diploma/Secondary', 1: 'Bachelor', 2: 'Master', 3: 'Ph.D.'},
    'Years_Experience'                      : {0: '1–5 yrs', 1: '6–10 yrs', 2: '11+ yrs'},
    'Years_experience_lastorganization'     : {0: '1–5 yrs', 1: '6–10 yrs', 2: '11+ yrs'},
    'Training_programs_During_last_three_years': {0: 'No Trainings', 1: '1–3', 2: '4–6', 3: '7+'},
}

# Feature-specific HR actions triggered by SHAP direction and feature value
def get_feature_recommendations(employee_series: pd.Series,
                                  shap_row: np.ndarray,
                                  top_n: int = 5) -> list:
    """
    Returns a list of actionable HR recommendations based on
    the top SHAP-driving features for this employee.

    Logic:
    - Only considers features where SHAP > 0 (pushing toward leaving)
    - Maps feature + value → specific HR action
    """

    # Build ranked list of features pushing toward leaving
    shap_df = pd.DataFrame({
        'Feature' : X_test.columns,
        'Value'   : employee_series.values,
        'SHAP'    : shap_row
    })
    # Only features pushing toward leaving (SHAP > 0)
    risk_drivers = shap_df[shap_df['SHAP'] > 0].sort_values('SHAP', ascending=False).head(top_n)

    recommendations = []

    for _, row in risk_drivers.iterrows():
        feat  = row['Feature']
        val   = int(round(row['Value'])) if row['Value'] == round(row['Value']) else row['Value']
        shap_ = row['SHAP']

        # Get human-readable value label
        label = FEATURE_LABELS.get(feat, {}).get(val, str(val))

        action = _get_action(feat, val)
        if action:
            recommendations.append({
                'feature'    : feat,
                'value_label': label,
                'shap_score' : round(shap_, 4),
                'action'     : action,
                'priority'   : 'URGENT' if shap_ > 0.15 else 'RECOMMENDED'
            })

    return recommendations


# =============================================================
# SECTION 4.1 — LEVEL 3 AGENTIC INTERVENTION TOOLS
# =============================================================

def get_market_salary_benchmark(academic_degree: int, years_experience: int) -> str:
    """
    Looks up regional salary average benchmarks in Saudi Riyals (SAR) for a given academic degree level and years of experience.
    academic_degree mappings: 0=Diploma/Secondary, 1=Bachelor, 2=Master, 3=Ph.D.
    years_experience mappings: 0=1-5 yrs, 1=6-10 yrs, 2=11+ yrs
    """
    degree_map = {0: 'Diploma/Secondary', 1: 'Bachelor', 2: 'Master', 3: 'Ph.D.'}
    exp_map = {0: '1-5 yrs', 1: '6-10 yrs', 2: '11+ yrs'}
    
    deg = degree_map.get(int(academic_degree), "Bachelor")
    exp = exp_map.get(int(years_experience), "1-5 yrs")
    
    benchmarks = {
        ('Diploma/Secondary', '1-5 yrs'): "5,000 - 7,000 SAR",
        ('Diploma/Secondary', '6-10 yrs'): "7,500 - 10,000 SAR",
        ('Diploma/Secondary', '11+ yrs'): "10,500 - 14,000 SAR",
        ('Bachelor', '1-5 yrs'): "8,000 - 12,000 SAR",
        ('Bachelor', '6-10 yrs'): "13,000 - 18,000 SAR",
        ('Bachelor', '11+ yrs'): "19,000 - 25,000 SAR",
        ('Master', '1-5 yrs'): "11,000 - 15,000 SAR",
        ('Master', '6-10 yrs'): "16,000 - 22,000 SAR",
        ('Master', '11+ yrs'): "23,000 - 32,000 SAR",
        ('Ph.D.', '1-5 yrs'): "15,000 - 20,000 SAR",
        ('Ph.D.', '6-10 yrs'): "21,000 - 28,000 SAR",
        ('Ph.D.', '11+ yrs'): "29,000 - 40,000+ SAR",
    }
    
    range_str = benchmarks.get((deg, exp), "8,000 - 12,000 SAR")
    return f"Saudi Regional average salary benchmark for education level '{deg}' and experience range '{exp}' is: {range_str}."

def get_tailored_training_modules(academic_degree: int, years_experience: int) -> str:
    """
    Retrieves recommended professional corporate training courses tailored to the employee's education and experience tier.
    academic_degree mappings: 0=Diploma/Secondary, 1=Bachelor, 2=Master, 3=Ph.D.
    years_experience mappings: 0=1-5 yrs, 1=6-10 yrs, 2=11+ yrs
    """
    degree_map = {0: 'Diploma/Secondary', 1: 'Bachelor', 2: 'Master', 3: 'Ph.D.'}
    exp_map = {0: '1-5 yrs', 1: '6-10 yrs', 2: '11+ yrs'}
    
    deg = degree_map.get(int(academic_degree), "Bachelor")
    exp = exp_map.get(int(years_experience), "1-5 yrs")
    
    if exp == '1-5 yrs':
        return "Recommended Training: [Technical Upskilling Bootcamps, Workplace Efficiency Foundations, Agile Development Practices]."
    elif exp == '6-10 yrs':
        return "Recommended Training: [Middle Management Leadership Program, Strategic Execution, High-Performance Communication Workshops]."
    else:
        return "Recommended Training: [Executive Decision-Making Seminar, Advanced Corporate Strategy, Organizational Growth Mentoring]."

def calculate_promotion_index(years_experience_lastorganization: int) -> str:
    """
    Computes an eligibility indicator score for career growth/promotions based on consecutive years at the current organization.
    years_experience_lastorganization mappings: 0=1-5 yrs, 1=6-10 yrs, 2=11+ yrs
    """
    exp_org_map = {0: '1-5 yrs', 1: '6-10 yrs', 2: '11+ yrs'}
    exp = exp_org_map.get(int(years_experience_lastorganization), "1-5 yrs")
    
    if exp == '1-5 yrs':
        return "Promotion Eligibility Index: 65% (Standard development cycle, review performance ratings next)."
    elif exp == '6-10 yrs':
        return "Promotion Eligibility Index: 88% (Highly eligible, standard review cycle exceeded. Immediate career progression path recommended)."
    else:
        return "Promotion Eligibility Index: 95% (Critical tenure benchmark. Retention threat if promotion is neglected)."


# =============================================================
# SECTION 4.2 — COGNITIVE REASONING CLIENT (LEVEL 3 AGENT)
# =============================================================

def get_agentic_recommendations_level3(employee_series: pd.Series, shap_row: np.ndarray, top_n: int = 5) -> dict:
    """
    Uses Gemini REST API and registered function tools to dynamically formulate a personalized multi-step HR intervention plan.
    """
    # Build list of top drivers pushing toward leaving (SHAP > 0)
    shap_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Value': employee_series.values,
        'SHAP': shap_row
    })
    risk_drivers = shap_df[shap_df['SHAP'] > 0].sort_values('SHAP', ascending=False).head(top_n)
    
    # Natural language features list
    feature_profile = {}
    for col, val in employee_series.items():
        if col in FEATURE_LABELS:
            lbl = FEATURE_LABELS[col].get(int(round(val)), str(val))
            feature_profile[col] = lbl
        else:
            feature_profile[col] = str(val)
            
    drivers_list = []
    for _, row in risk_drivers.iterrows():
        feat = row['Feature']
        val = int(round(row['Value'])) if row['Value'] == round(row['Value']) else row['Value']
        lbl = FEATURE_LABELS.get(feat, {}).get(val, str(val))
        drivers_list.append(f"- {feat.replace('_', ' ')}: Currently '{lbl}' (SHAP score: {row['SHAP']:.4f})")
        
    drivers_str = "\n".join(drivers_list)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not configured.")
        
    prompt = f"""
Employee Profile Context:
- Academic Degree: {feature_profile.get('Academic_degree', 'Unknown')}
- Total Experience: {feature_profile.get('Years_Experience', 'Unknown')}
- Years at Current Organization: {feature_profile.get('Years_experience_lastorganization', 'Unknown')}
- Current Salary Bracket: {feature_profile.get('MonthlySalary', 'Unknown')}
- Overtime frequency: {feature_profile.get('OverTime', 'Unknown')}
- Overtime payment: {feature_profile.get('Payment_Overtime', 'Unknown')}
- Deserved promotion status: {feature_profile.get('Get_Deserved_Promotion', 'Unknown')}

Top Attrition Drivers (SHAP Indicators):
{drivers_str}

Please generate an expert HR intervention plan. You MUST run the appropriate lookup tools.

Write a structured response with:
1. "thought_process": Summarize your reasoning and root-cause analysis based on features and tool outputs.
2. "recommendations": A list of 2 to 4 structured actionable steps. Each step must have:
   - "feature": The main risk feature addressed.
   - "value_label": Current status.
   - "action": Highly specific intervention based on your tool checks.
   - "priority": 'URGENT' or 'RECOMMENDED'.
"""

    sys_instruct = """
You are a Saudi Attrition & Retention Intelligence Agent, an expert in human resource strategy and employee engagement.
Analyze the employee profile, execute the required tools to gather context about regional benchmarks, promotion indices, and training courses, and synthesize a professional multi-step plan to retain the employee.
"""

    # Gemini API tool formatting
    tools = [{
        "functionDeclarations": [
            {
                "name": "get_market_salary_benchmark",
                "description": "Looks up regional salary average benchmarks in Saudi Riyals (SAR) for a given academic degree level and years of experience.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "academic_degree": {"type": "INTEGER"},
                        "years_experience": {"type": "INTEGER"}
                    },
                    "required": ["academic_degree", "years_experience"]
                }
            },
            {
                "name": "get_tailored_training_modules",
                "description": "Retrieves recommended professional corporate training courses tailored to the employee's education and experience tier.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "academic_degree": {"type": "INTEGER"},
                        "years_experience": {"type": "INTEGER"}
                    },
                    "required": ["academic_degree", "years_experience"]
                }
            },
            {
                "name": "calculate_promotion_index",
                "description": "Computes an eligibility indicator score for career growth/promotions based on consecutive years at the current organization.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "years_experience_lastorganization": {"type": "INTEGER"}
                    },
                    "required": ["years_experience_lastorganization"]
                }
            }
        ]
    }]

    available_tools = {
        "get_market_salary_benchmark": get_market_salary_benchmark,
        "get_tailored_training_modules": get_tailored_training_modules,
        "calculate_promotion_index": calculate_promotion_index
    }

    import requests
    from tenacity import retry, stop_after_attempt, wait_random_exponential
    
    @retry(
        wait=wait_random_exponential(min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def call_gemini():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
        
        payload = {
            "systemInstruction": {
                "parts": [{"text": sys_instruct}]
            },
            "contents": contents,
            "tools": tools,
            "generationConfig": {
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        res_data = response.json()
        
        # Check if model wants to call a tool
        try:
            content = res_data['candidates'][0]['content']
            parts = content.get('parts', [])
            
            func_call_part = next((p for p in parts if 'functionCall' in p), None)
            if func_call_part:
                func_call = func_call_part['functionCall']
                function_name = func_call['name']
                function_args = func_call.get('args', {})
                
                # Execute tool
                function_to_call = available_tools[function_name]
                if function_name == "calculate_promotion_index":
                    function_response = function_to_call(
                        years_experience_lastorganization=int(function_args.get("years_experience_lastorganization", 0))
                    )
                else:
                    function_response = function_to_call(
                        academic_degree=int(function_args.get("academic_degree", 0)),
                        years_experience=int(function_args.get("years_experience", 0))
                    )
                
                # Append model's tool call exactly as received
                contents.append(content)
                
                # Append tool response
                contents.append({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": function_name,
                            "response": {"result": function_response}
                        }
                    }]
                })
                
                # Explicitly prompt for JSON output
                contents.append({
                    "role": "user",
                    "parts": [{"text": "Tool executed successfully. Now provide the final response in JSON format according to the requested schema."}]
                })
                
                # Make second request
                payload["contents"] = contents
                second_response = requests.post(url, headers=headers, json=payload)
                second_response.raise_for_status()
                sec_parts = second_response.json()['candidates'][0]['content'].get('parts', [])
                return "".join(p.get('text', '') for p in sec_parts)
            else:
                return "".join(p.get('text', '') for p in parts) or '{}'
        except (KeyError, IndexError) as e:
            return '{}'
            
    response_text = call_gemini()
    
    try:
        data = json.loads(response_text)
    except Exception:
        import re
        data = None
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                pass
        if not data:
            data = {
                "thought_process": f"Failed parsing JSON from model output. Raw output: {response_text}",
                "recommendations": []
            }
        
        # Ensure thought_process exists
        if "thought_process" not in data or not data["thought_process"]:
            data["thought_process"] = "The Advisory Engine generated these recommendations based on a strategic review of the employee's churn drivers."
            
    # Inject actual SHAP scores to ensure downstream dashboard compatibility
    recs = data.get("recommendations", [])
    if isinstance(recs, list):
        for r in recs:
            if isinstance(r, dict):
                feat_name = r.get("feature", "")
                # Clean up if model returned space instead of underscore
                if feat_name not in X_test.columns:
                    # Try matching with underscore replacement
                    cleaned_feat = feat_name.replace(" ", "_")
                    if cleaned_feat in X_test.columns:
                        r['feature'] = cleaned_feat
                        feat_name = cleaned_feat
                        
                actual_shap = shap_df.loc[shap_df['Feature'] == feat_name, 'SHAP']
                if not actual_shap.empty:
                    r['shap_score'] = float(actual_shap.values[0])
                else:
                    r['shap_score'] = 0.0
                    
    return data


def _get_action(feature: str, value) -> str:
    """
    Maps a specific feature + encoded value to an HR action string.
    Returns None if no action is warranted.
    """
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


# =============================================================
# SECTION 5 — EMAIL SIMULATION FUNCTION
# =============================================================

def simulate_email(employee_id: int,
                   risk_level: str,
                   probability: float,
                   recommendations: list,
                   actual_label: int) -> str:
    """
    Simulates an HR email notification.
    In production: replace with smtplib or SendGrid API call.
    Returns the email body as a string.
    """
    now        = datetime.now().strftime("%Y-%m-%d %H:%M")
    emoji_map  = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
    emoji      = emoji_map.get(risk_level, '')
    actual_str = "LEFT ✗" if actual_label == 1 else "STAYED ✓"

    lines = [
        f"FROM    : {SENDER}",
        f"TO      : {HR_EMAIL}",
        f"SUBJECT : {emoji} [{risk_level} RISK] Employee #{employee_id} — Churn Alert",
        f"DATE    : {now}",
        "=" * 60,
        "",
        f"Dear HR Team,",
        "",
        f"The Churn Prediction Agent has flagged Employee #{employee_id}.",
        "",
        f"  Risk Level        : {emoji} {risk_level}",
        f"  Churn Probability : {probability:.1%}",
        f"  Actual Outcome    : {actual_str}  (ground truth from test set)",
        "",
        "─" * 60,
        "TOP RISK DRIVERS & RECOMMENDED ACTIONS",
        "─" * 60,
    ]

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            lines += [
                f"",
                f"  [{rec['priority']}] {i}. {rec['feature'].replace('_', ' ')}",
                f"     Current Value : {rec['value_label']}",
                f"     SHAP Score    : {rec['shap_score']:+.4f}",
                f"     Action        : {rec['action']}",
            ]
    else:
        lines.append("  No specific feature-based actions identified.")

    lines += [
        "",
        "─" * 60,
        "GENERAL RECOMMENDATION",
        "─" * 60,
        "",
    ]

    if risk_level == 'CRITICAL':
        lines.append("  ⚡ Schedule an IMMEDIATE retention meeting with this employee.")
        lines.append("  ⚡ Escalate to department head and senior HR within 24 hours.")
    elif risk_level == 'HIGH':
        lines.append("  ⚠️  Schedule a retention conversation within this week.")
        lines.append("  ⚠️  Review compensation and growth path.")
    elif risk_level == 'MEDIUM':
        lines.append("  📋 Add to monthly HR monitoring list.")
        lines.append("  📋 Check in with direct manager about employee engagement.")
    else:
        lines.append("  ✅ No immediate action required. Continue regular monitoring.")

    lines += [
        "",
        "─" * 60,
        "This is an automated message from the Employee Churn Prediction Agent.",
        "Do not reply to this email.",
        "=" * 60,
    ]

    return "\n".join(lines)


# =============================================================
# SECTION 6 — RUN AGENT ON ALL TEST EMPLOYEES
# =============================================================
print()
print("=" * 60)
print("SECTION 6 — RUNNING AGENT ON ALL TEST EMPLOYEES")
print("=" * 60)

# Load classification threshold if optimized in Phase 3
threshold_path = os.path.join(OUTPUT_DIR, 'best_threshold.txt')
if os.path.exists(threshold_path):
    with open(threshold_path, 'r') as f:
        CLASSIFICATION_THRESHOLD = float(f.read().strip())
else:
    CLASSIFICATION_THRESHOLD = 0.50
print(f"✅ Loaded classification threshold: {CLASSIFICATION_THRESHOLD:.2f}")

probabilities = model.predict_proba(X_test)[:, 1]
predictions   = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

# Aggregate results
agent_results = []

for i in range(len(X_test)):
    prob            = probabilities[i]
    risk_level, emoji, desc = classify_risk(prob)
    employee_series = X_test.iloc[i]
    shap_row        = sv[i]
    actual          = y_test.iloc[i]

    recommendations = get_feature_recommendations(employee_series, shap_row, top_n=5)

    agent_results.append({
        'employee_index' : X_test.index[i],
        'churn_prob'     : round(prob, 4),
        'risk_level'     : risk_level,
        'predicted'      : int(predictions[i]),
        'actual'         : int(actual),
        'correct'        : int(predictions[i]) == int(actual),
        'num_actions'    : len(recommendations),
        'top_driver'     : recommendations[0]['feature'] if recommendations else 'N/A',
    })

results_df = pd.DataFrame(agent_results)

# Summary
print(f"\n{'Risk Level':<12} {'Count':>8} {'% of Test':>12}")
print("-" * 35)
for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    count = (results_df['risk_level'] == level).sum()
    pct   = count / len(results_df) * 100
    emoji_map = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
    print(f"{emoji_map[level]} {level:<10} {count:>8} {pct:>11.1f}%")

print(f"\n   Total employees assessed : {len(results_df)}")
print(f"   Flagged (Medium+)        : {(results_df['risk_level'] != 'LOW').sum()}")
print(f"   Critical alerts          : {(results_df['risk_level'] == 'CRITICAL').sum()}")

# Save full results
results_path = os.path.join(OUTPUT_DIR, 'agent_risk_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\n✅ Full results saved: agent_risk_results.csv")

# =============================================================
# SECTION 7 — GENERATE DETAILED REPORTS + EMAILS
#             (Critical & High Risk only — to avoid noise)
# =============================================================
print()
print("=" * 60)
print("SECTION 7 — GENERATING REPORTS & SIMULATED EMAILS")
print("           (Critical + High Risk employees only)")
print("=" * 60)

flagged = results_df[results_df['risk_level'].isin(['CRITICAL', 'HIGH'])].copy()
print(f"\n   Generating reports for {len(flagged)} employees...")

email_log = []

for _, row in flagged.iterrows():
    idx             = row['employee_index']
    i               = X_test.index.get_loc(idx)
    prob            = probabilities[i]
    risk_level, _, _ = classify_risk(prob)
    employee_series = X_test.iloc[i]
    shap_row        = sv[i]
    actual          = y_test.iloc[i]

    recommendations = get_feature_recommendations(employee_series, shap_row, top_n=5)

    # Generate email
    email_body = simulate_email(
        employee_id     = idx,
        risk_level      = risk_level,
        probability     = prob,
        recommendations = recommendations,
        actual_label    = int(actual)
    )

    # Save individual email/report to file
    report_filename = f"report_employee_{idx}_{risk_level}.txt"
    report_path     = os.path.join(REPORT_DIR, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(email_body)

    email_log.append({
        'employee_index': idx,
        'risk_level'    : risk_level,
        'churn_prob'    : round(prob, 4),
        'report_file'   : report_filename,
        'email_sent_to' : HR_EMAIL,
        'timestamp'     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

# Save email log
email_log_df = pd.DataFrame(email_log)
email_log_df.to_csv(os.path.join(OUTPUT_DIR, 'email_log.csv'), index=False)
print(f"\n✅ {len(flagged)} reports saved to: outputs/agent_reports/")
print(f"✅ Email log saved: email_log.csv")

# =============================================================
# SECTION 8 — PRINT SAMPLE REPORT (First Critical Employee)
# =============================================================
print()
print("=" * 60)
print("SECTION 8 — SAMPLE REPORT (First Critical Risk Employee)")
print("=" * 60)

critical_employees = results_df[results_df['risk_level'] == 'CRITICAL']

if len(critical_employees) > 0:
    sample_row      = critical_employees.iloc[0]
    sample_idx      = sample_row['employee_index']
    i               = X_test.index.get_loc(sample_idx)
    prob            = probabilities[i]
    risk_level, _, _ = classify_risk(prob)
    employee_series = X_test.iloc[i]
    shap_row        = sv[i]
    actual          = y_test.iloc[i]

    recommendations = get_feature_recommendations(employee_series, shap_row, top_n=5)
    email_body      = simulate_email(
        employee_id     = sample_idx,
        risk_level      = risk_level,
        probability     = prob,
        recommendations = recommendations,
        actual_label    = int(actual)
    )
    print()
    print(email_body)
else:
    print("   No CRITICAL risk employees found in test set.")
    # Show first HIGH risk instead
    high_employees = results_df[results_df['risk_level'] == 'HIGH']
    if len(high_employees) > 0:
        sample_row      = high_employees.iloc[0]
        sample_idx      = sample_row['employee_index']
        i               = X_test.index.get_loc(sample_idx)
        prob            = probabilities[i]
        risk_level, _, _ = classify_risk(prob)
        employee_series = X_test.iloc[i]
        shap_row        = sv[i]
        actual          = y_test.iloc[i]

        recommendations = get_feature_recommendations(employee_series, shap_row, top_n=5)
        email_body      = simulate_email(
            employee_id     = sample_idx,
            risk_level      = risk_level,
            probability     = prob,
            recommendations = recommendations,
            actual_label    = int(actual)
        )
        print()
        print(email_body)

# =============================================================
# SECTION 9 — AGENT ACCURACY SUMMARY
# =============================================================
print()
print("=" * 60)
print("SECTION 9 — AGENT ACCURACY BY RISK LEVEL")
print("=" * 60)

print(f"\n{'Risk Level':<12} {'Total':>8} {'Correct':>10} {'Accuracy':>12}")
print("-" * 45)

for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    subset   = results_df[results_df['risk_level'] == level]
    if len(subset) == 0:
        continue
    correct  = subset['correct'].sum()
    total    = len(subset)
    accuracy = correct / total * 100
    emoji_map = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
    print(f"{emoji_map[level]} {level:<10} {total:>8} {correct:>10} {accuracy:>11.1f}%")

overall_acc = results_df['correct'].mean() * 100
print(f"\n   Overall Agent Accuracy : {overall_acc:.1f}%")

# =============================================================
# FINAL SUMMARY
# =============================================================
print()
print("=" * 60)
print("PHASE 4 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
Agent           : Employee Churn Risk Agent
Employees       : {len(results_df)} assessed
Risk Levels     : CRITICAL (≥80%) / HIGH (≥60%) / MEDIUM (≥40%) / LOW (<40%)
Flagged         : {(results_df['risk_level'] != 'LOW').sum()} employees (Medium or above)
Critical Alerts : {(results_df['risk_level'] == 'CRITICAL').sum()} employees
Emails Sent     : {len(flagged)} (simulated — Critical + High only)
Overall Accuracy: {overall_acc:.1f}%

Recommendation Engine:
  → SHAP-driven: top 5 risk-driving features per employee
  → Feature-specific HR actions (salary, stress, promotion, etc.)
  → Priority flags: URGENT (SHAP > 0.15) / RECOMMENDED

Files saved to outputs/:
  ✅ agent_risk_results.csv      ← Full risk table for all test employees
  ✅ email_log.csv               ← Log of all simulated emails sent
  ✅ agent_reports/              ← Individual report per Critical/High employee

Next Step: Run 5_notification.py (live email/Slack integration)
          or 6_dashboard.py (Streamlit HR dashboard)
""")