============================================================
EMPLOYEE CHURN PREDICTION PROJECT
Saudi Employee Attrition Dataset
============================================================

FOLDER STRUCTURE:
-----------------
employee_churn_project\
    │
    ├── data\                        ← PUT YOUR EXCEL FILE HERE
    │    └── Employee_attrition_dataset_for_tree-based_models.xlsx
    │
    ├── outputs\                     ← All plots will be saved here automatically
    │
    ├── 1_data_exploration.py        ← Phase 1 (ready to run)
    ├── 2_model_training.py          ← Phase 2 (coming soon)
    ├── 3_model_evaluation.py        ← Phase 3 (coming soon)
    ├── 4_agent_logic.py             ← Phase 4 (coming soon)
    ├── 5_notification.py            ← Phase 5 (coming soon)
    ├── 6_dashboard.py               ← Phase 6 (coming soon)
    └── README.txt                   ← This file

============================================================
SETUP INSTRUCTIONS (Run these ONCE in terminal)
============================================================

1. Create virtual environment:
   python -m venv churn_env

2. Activate it (Windows):
   churn_env\Scripts\activate

3. Install all libraries:
   pip install pandas matplotlib seaborn scikit-learn xgboost lightgbm openpyxl shap streamlit

============================================================
HOW TO RUN (Every session)
============================================================

1. Activate virtual environment:
   churn_env\Scripts\activate

2. Run Phase 1:
   python 1_data_exploration.py

============================================================
