# =============================================================
# PHASE 2 — MODEL TRAINING
# Project : Employee Churn Prediction
# Dataset : Saudi Employee Attrition Dataset
# Models  : Logistic Regression, Random Forest,
#           XGBoost, LightGBM, CatBoost
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics         import (accuracy_score, f1_score,
                                     roc_auc_score, recall_score,
                                     precision_score)
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from catboost                import CatBoostClassifier

# =============================================================
# CONFIGURATION
# =============================================================
DATA_PATH  = r"data\Employee_attrition_dataset_for_tree-based_models.xlsx"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# =============================================================
# SECTION 1 — LOAD DATA
# =============================================================
print("=" * 60)
print("SECTION 1 — LOADING DATA")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Separate features and target
X = df.drop(columns=['Attrition'])
y = df['Attrition']

print(f"\n   Features (X) : {X.shape[1]} columns")
print(f"   Target   (y) : Attrition (0=Stayed, 1=Left)")
print(f"   Class split  : {y.value_counts()[0]} stayed | {y.value_counts()[1]} left")

# =============================================================
# SECTION 2 — TRAIN / TEST SPLIT
# =============================================================
print()
print("=" * 60)
print("SECTION 2 — TRAIN / TEST SPLIT (80% / 20%)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y        # keeps same 0/1 ratio in both splits
)

print(f"\n   Training set : {X_train.shape[0]} rows")
print(f"   Testing set  : {X_test.shape[0]} rows")
print(f"\n   Train target split : {y_train.value_counts()[0]} stayed | {y_train.value_counts()[1]} left")
print(f"   Test  target split : {y_test.value_counts()[0]} stayed  | {y_test.value_counts()[1]} left")
print(f"\n✅ Stratified split confirmed — same ratio in both sets")

# =============================================================
# SECTION 3 — DEFINE 5 MODELS
# =============================================================
print()
print("=" * 60)
print("SECTION 3 — DEFINING 5 MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        random_state = RANDOM_STATE,
        max_iter     = 1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators = 100,
        random_state = RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators        = 100,
        random_state        = RANDOM_STATE,
        eval_metric         = 'logloss',
        verbosity           = 0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators = 100,
        random_state = RANDOM_STATE,
        verbose      = -1
    ),
    "CatBoost": CatBoostClassifier(
        iterations   = 100,
        random_seed  = RANDOM_STATE,
        verbose      = 0
    )
}

print(f"\n✅ {len(models)} models defined and ready to train")
for name in models:
    print(f"   → {name}")

# =============================================================
# SECTION 4 — CROSS VALIDATION + TRAINING
# =============================================================
print()
print("=" * 60)
print("SECTION 4 — TRAINING & CROSS VALIDATION (5-Fold)")
print("=" * 60)

cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results  = {}
scoring  = ['accuracy', 'f1', 'roc_auc', 'recall', 'precision']

for name, model in models.items():
    print(f"\n   Training: {name} ...")

    # Cross validation
    cv_scores = cross_validate(
        model, X_train, y_train,
        cv            = cv,
        scoring       = scoring,
        return_train_score = False
    )

    # Train on full training set
    model.fit(X_train, y_train)

    # Test set predictions
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model'       : model,
        'cv_accuracy' : cv_scores['test_accuracy'].mean(),
        'cv_f1'       : cv_scores['test_f1'].mean(),
        'cv_roc_auc'  : cv_scores['test_roc_auc'].mean(),
        'cv_recall'   : cv_scores['test_recall'].mean(),
        'cv_precision': cv_scores['test_precision'].mean(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_f1'      : f1_score(y_test, y_pred),
        'test_roc_auc' : roc_auc_score(y_test, y_pred_prob),
        'test_recall'  : recall_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
    }

    print(f"   ✅ Done — CV F1: {results[name]['cv_f1']:.3f} | Test F1: {results[name]['test_f1']:.3f}")

# =============================================================
# SECTION 5 — COMPARISON TABLE
# =============================================================
print()
print("=" * 60)
print("SECTION 5 — MODEL COMPARISON RESULTS")
print("=" * 60)

comparison_df = pd.DataFrame({
    name: {
        'CV Accuracy'  : round(v['cv_accuracy'],  3),
        'CV F1 Score'  : round(v['cv_f1'],        3),
        'CV ROC-AUC'   : round(v['cv_roc_auc'],   3),
        'CV Recall'    : round(v['cv_recall'],     3),
        'CV Precision' : round(v['cv_precision'],  3),
        'Test Accuracy': round(v['test_accuracy'], 3),
        'Test F1 Score': round(v['test_f1'],       3),
        'Test ROC-AUC' : round(v['test_roc_auc'],  3),
        'Test Recall'  : round(v['test_recall'],   3),
        'Test Precision': round(v['test_precision'],3),
    }
    for name, v in results.items()
}).T

print("\n--- Cross Validation Scores (5-Fold) ---")
print(comparison_df[['CV Accuracy','CV F1 Score','CV ROC-AUC','CV Recall','CV Precision']].to_string())

print("\n--- Test Set Scores ---")
print(comparison_df[['Test Accuracy','Test F1 Score','Test ROC-AUC','Test Recall','Test Precision']].to_string())

# Save comparison to CSV
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))
print("\n✅ Comparison saved: model_comparison.csv")

# =============================================================
# SECTION 6 — PICK BEST MODEL (by Test F1 Score)
# =============================================================
print()
print("=" * 60)
print("SECTION 6 — SELECTING BEST MODEL")
print("=" * 60)

best_name  = comparison_df['Test F1 Score'].idxmax()
best_score = comparison_df.loc[best_name, 'Test F1 Score']
best_model = results[best_name]['model']

print(f"\n🏆 Best Model  : {best_name}")
print(f"   Test F1     : {best_score:.3f}")
print(f"   Test ROC-AUC: {comparison_df.loc[best_name, 'Test ROC-AUC']:.3f}")
print(f"   Test Recall : {comparison_df.loc[best_name, 'Test Recall']:.3f}")
print(f"   Test Accuracy: {comparison_df.loc[best_name, 'Test Accuracy']:.3f}")

# Save best model
model_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✅ Best model saved: best_model.pkl")

# Save model name
name_path = os.path.join(OUTPUT_DIR, 'best_model_name.txt')
with open(name_path, 'w') as f:
    f.write(best_name)
print(f"✅ Model name saved: best_model_name.txt")

# Save test sets for Phase 3
test_data_path = os.path.join(OUTPUT_DIR, 'test_data.pkl')
with open(test_data_path, 'wb') as f:
    pickle.dump({'X_test': X_test, 'y_test': y_test,
                 'X_train': X_train, 'y_train': y_train}, f)
print(f"✅ Test data saved : test_data.pkl")

# =============================================================
# SECTION 7 — PLOT: MODEL COMPARISON BAR CHART
# =============================================================
print()
print("=" * 60)
print("SECTION 7 — GENERATING PLOTS")
print("=" * 60)

metrics    = ['Test Accuracy', 'Test F1 Score', 'Test ROC-AUC', 'Test Recall']
model_names = list(results.keys())
x          = np.arange(len(metrics))
width      = 0.15
colors     = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

fig, ax = plt.subplots(figsize=(14, 7))

for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [comparison_df.loc[name, m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison — All 5 Models', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=10)
ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.4, label='0.8 threshold')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_model_comparison.png'), dpi=150)
plt.show()
print("✅ Saved: plot_model_comparison.png")

# =============================================================
# SECTION 8 — FEATURE IMPORTANCE (Best Model)
# =============================================================
print()
print("=" * 60)
print(f"SECTION 8 — FEATURE IMPORTANCE ({best_name})")
print("=" * 60)

# Get feature importance if available
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature'   : X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_feat = plt.cm.RdYlGn_r(
        np.linspace(0.1, 0.9, len(feat_df))
    )
    bars = ax.barh(
        feat_df['Feature'][::-1],
        feat_df['Importance'][::-1],
        color=colors_feat[::-1],
        edgecolor='white'
    )
    for bar, val in zip(bars, feat_df['Importance'][::-1]):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(f'Top 15 Features — {best_name}',
                 fontsize=13, fontweight='bold')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_feature_importance.png'), dpi=150)
    plt.show()
    print("✅ Saved: plot_feature_importance.png")

    print("\nTop 15 Most Important Features:")
    for i, row in feat_df.iterrows():
        print(f"   {feat_df.index.get_loc(i)+1:02d}. {row['Feature']:<45} | {row['Importance']:.4f}")

elif best_name == "Logistic Regression":
    coef_df = pd.DataFrame({
        'Feature'    : X.columns,
        'Coefficient': np.abs(best_model.coef_[0])
    }).sort_values('Coefficient', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(coef_df['Feature'][::-1], coef_df['Coefficient'][::-1],
            color='#2196F3', edgecolor='white')
    ax.set_title('Top 15 Features — Logistic Regression (Coefficients)',
                 fontsize=13, fontweight='bold')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_feature_importance.png'), dpi=150)
    plt.show()
    print("✅ Saved: plot_feature_importance.png")

# =============================================================
# FINAL SUMMARY
# =============================================================
print()
print("=" * 60)
print("PHASE 2 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
Models Trained : 5
Best Model     : {best_name}
Best F1 Score  : {best_score:.3f}
Best ROC-AUC   : {comparison_df.loc[best_name, 'Test ROC-AUC']:.3f}

Files saved to outputs/:
  ✅ best_model.pkl          ← Used in Phase 3 & 4
  ✅ best_model_name.txt     ← Model name reference
  ✅ test_data.pkl           ← Used in Phase 3
  ✅ model_comparison.csv    ← All scores spreadsheet
  ✅ plot_model_comparison.png
  ✅ plot_feature_importance.png

Next Step: Run 3_model_evaluation.py
""")
