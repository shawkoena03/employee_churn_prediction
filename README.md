# 🏢 Employee Churn Prediction & Retention Agent System

An end-to-end Machine Learning and AI-driven Retention Pipeline designed to predict employee attrition, analyze root causes using SHAP explainability, generate personalized retention strategies via LLM Agents (Gemini), and alert HR via automated email workflows and interactive Streamlit dashboards.

---

## 📌 Project Overview

- **Dataset**: Saudi Employee Attrition Dataset (Tree-based Model Optimized)
- **Objective**: Proactively identify at-risk talent, understand individual churn drivers, and deliver targeted intervention plans.
- **Key Modules**:
  1. **EDA & Visualizations** (`1_data_exploration.py`): In-depth demographic, satisfaction, stress, and salary correlation analysis.
  2. **Model Training & Hyperparameter Tuning** (`2_model_training.py`): Comparative evaluation of CatBoost, LightGBM, XGBoost, and Random Forest with Optuna Bayesian optimization.
  3. **Evaluation & Explainability** (`3_model_evaluation.py`): ROC-AUC, Precision-Recall curves, confusion matrices, and SHAP value explanations.
  4. **Agentic Retention Logic** (`4_agentic_logic.py`): Multi-tier risk categorization (Low, Medium, High, Critical) and LLM-powered personalized retention memos.
  5. **Notification Pipeline** (`5_notification.py`): Automated HTML email generation and SMTP delivery to HR partners.
  6. **Executive Dashboard** (`6_dashboard.py`): Interactive Streamlit command center for workforce analytics, risk scanning, and simulation.

---

## 📂 Project Structure

```text
employee_churn_project/
├── data/
│   └── Employee_attrition_dataset_for_tree-based_models.xlsx
├── outputs/
│   ├── agent_reports/         # LLM retention recommendations per employee
│   ├── simulated_emails/      # Formatted HR warning emails
│   ├── *.png                  # EDA and Model Evaluation charts
│   ├── best_model_tuned.pkl   # Production-ready model
│   └── agent_risk_results.csv # Risk assessment scores
├── 1_data_exploration.py
├── 2_model_training.py
├── 3_model_evaluation.py
├── 4_agentic_logic.py
├── 5_notification.py
├── 6_dashboard.py
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv churn_env
churn_env\Scripts\activate

# Linux / macOS
python3 -m venv churn_env
source churn_env/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna shap matplotlib seaborn streamlit openpyxl python-dotenv google-genai
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Update your keys:
- `GEMINI_API_KEY`: Google Gemini API key for retention strategy generation.
- `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`: SMTP email settings for HR notifications.

---

## 🛠️ Execution Pipeline

Run the pipeline steps sequentially:

```bash
# Phase 1: Exploratory Data Analysis
python 1_data_exploration.py

# Phase 2: Model Training & Tuning (Optuna)
python 2_model_training.py

# Phase 3: Model Evaluation & SHAP Analysis
python 3_model_evaluation.py

# Phase 4: Agentic Retention Analysis
python 4_agentic_logic.py

# Phase 5: Notification Service
python 5_notification.py

# Phase 6: Launch Interactive Streamlit Dashboard
streamlit run 6_dashboard.py
```

---

## 📊 Dashboard Preview

Launch the Streamlit app to explore:
- High-level workforce health KPIs & Attrition distribution.
- Real-time churn probability inference.
- Automated generation of employee-specific retention memos.
- One-click HR email dispatch.

---

## 📄 License
This project is licensed under the MIT License.
