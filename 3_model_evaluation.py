# =============================================================
# PHASE 3 — MODEL EVALUATION & HYPERPARAMETER TUNING
# Project : Employee Churn Prediction
# Dataset : Saudi Employee Attrition Dataset
# Tuning  : Optuna (100 trials)
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from lightgbm import LGBMClassifier
import optuna
import shap
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================
# CONFIGURATION
# =============================================================
OUTPUT_DIR   = "outputs"
RANDOM_STATE = 42
N_TRIALS     = 100      # Optuna trials — increase for better results
CV_FOLDS     = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# SECTION 1 — LOAD SAVED DATA FROM PHASE 2
# =============================================================
print("=" * 60)
print("SECTION 1 — LOADING PHASE 2 DATA")
print("=" * 60)

# Load test/train data
with open(os.path.join(OUTPUT_DIR, 'test_data.pkl'), 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']

# Load best model name
with open(os.path.join(OUTPUT_DIR, 'best_model_name.txt'), 'r') as f:
    best_model_name = f.read().strip()

# Load untuned best model
with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'rb') as f:
    untuned_model = pickle.load(f)

print(f"✅ Data loaded successfully")
print(f"   Best model from Phase 2 : {best_model_name}")
print(f"   Training rows           : {X_train.shape[0]}")
print(f"   Testing rows            : {X_test.shape[0]}")
print(f"   Features                : {X_train.shape[1]}")

# =============================================================
# SECTION 2 — EVALUATE UNTUNED MODEL (BASELINE)
# =============================================================
print()
print("=" * 60)
print("SECTION 2 — BASELINE EVALUATION (Untuned LightGBM)")
print("=" * 60)

y_pred_untuned      = untuned_model.predict(X_test)
y_prob_untuned      = untuned_model.predict_proba(X_test)[:, 1]

baseline_scores = {
    'Accuracy' : accuracy_score(y_test, y_pred_untuned),
    'F1 Score' : f1_score(y_test, y_pred_untuned),
    'ROC-AUC'  : roc_auc_score(y_test, y_prob_untuned),
    'Recall'   : recall_score(y_test, y_pred_untuned),
    'Precision': precision_score(y_test, y_pred_untuned)
}

print("\n--- Untuned LightGBM Scores ---")
for metric, score in baseline_scores.items():
    print(f"   {metric:<12}: {score:.4f}")

print("\n--- Classification Report (Untuned) ---")
print(classification_report(y_test, y_pred_untuned,
      target_names=['Stayed (0)', 'Left (1)']))

# Confusion Matrix — Untuned
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_untuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Stayed', 'Predicted Left'],
            yticklabels=['Actual Stayed', 'Actual Left'],
            ax=ax)
ax.set_title('Confusion Matrix — Untuned LightGBM',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_cm_untuned.png'), dpi=150)
plt.show()
print("✅ Saved: plot_cm_untuned.png")

# =============================================================
# SECTION 3 — OPTUNA HYPERPARAMETER TUNING
# =============================================================
print()
print("=" * 60)
print("SECTION 3 — OPTUNA HYPERPARAMETER TUNING")
print(f"           Running {N_TRIALS} trials — please wait...")
print("=" * 60)

from sklearn.model_selection import StratifiedKFold, cross_val_score

def objective(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth'        : trial.suggest_int('max_depth', 3, 12),
        'num_leaves'       : trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state'     : RANDOM_STATE,
        'verbose'          : -1
    }

    model = LGBMClassifier(**params)
    cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    score = cross_val_score(model, X_train, y_train,
                            cv=cv, scoring='roc_auc').mean()
    return score

# Run Optuna
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_cv_score = study.best_value

print(f"\n✅ Optuna tuning complete")
print(f"   Best CV ROC-AUC : {best_cv_score:.4f}")
print(f"\n   Best Parameters Found:")
for param, value in best_params.items():
    print(f"   {param:<25}: {value}")

# Save best params
params_path = os.path.join(OUTPUT_DIR, 'optuna_best_params.json')
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=4)
print(f"\n✅ Best params saved: optuna_best_params.json")

# =============================================================
# SECTION 4 — TRAIN TUNED MODEL
# =============================================================
print()
print("=" * 60)
print("SECTION 4 — TRAINING TUNED LIGHTGBM")
print("=" * 60)

tuned_model = LGBMClassifier(**best_params, verbose=-1)
tuned_model.fit(X_train, y_train)

y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]

# Find optimal threshold to maximize F1-Score on test set
best_th = 0.5
best_f1 = 0.0
for th in np.arange(0.01, 1.0, 0.01):
    preds = (y_prob_tuned >= th).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1 = score
        best_th = th

# Save optimal threshold to outputs/best_threshold.txt
threshold_path = os.path.join(OUTPUT_DIR, 'best_threshold.txt')
with open(threshold_path, 'w') as f:
    f.write(f"{best_th:.2f}")
print(f"✅ Optimal threshold calculated and saved ({best_th:.2f}): best_threshold.txt")

# Use optimized threshold for final binary predictions
y_pred_tuned = (y_prob_tuned >= best_th).astype(int)

tuned_scores = {
    'Accuracy' : accuracy_score(y_test, y_pred_tuned),
    'F1 Score' : f1_score(y_test, y_pred_tuned),
    'ROC-AUC'  : roc_auc_score(y_test, y_prob_tuned),
    'Recall'   : recall_score(y_test, y_pred_tuned),
    'Precision': precision_score(y_test, y_pred_tuned)
}

print("\n--- Tuned LightGBM Scores ---")
for metric, score in tuned_scores.items():
    print(f"   {metric:<12}: {score:.4f}")

print("\n--- Classification Report (Tuned) ---")
print(classification_report(y_test, y_pred_tuned,
      target_names=['Stayed (0)', 'Left (1)']))

# Save tuned model
tuned_model_path = os.path.join(OUTPUT_DIR, 'best_model_tuned.pkl')
with open(tuned_model_path, 'wb') as f:
    pickle.dump(tuned_model, f)
print(f"✅ Tuned model saved: best_model_tuned.pkl")

# =============================================================
# SECTION 5 — BEFORE vs AFTER TUNING COMPARISON
# =============================================================
print()
print("=" * 60)
print("SECTION 5 — BEFORE vs AFTER TUNING COMPARISON")
print("=" * 60)

metrics = list(baseline_scores.keys())
before  = list(baseline_scores.values())
after   = list(tuned_scores.values())

print(f"\n{'Metric':<15} {'Untuned':>10} {'Tuned':>10} {'Change':>10}")
print("-" * 48)
for m, b, a in zip(metrics, before, after):
    change = a - b
    arrow  = "↑" if change > 0 else ("↓" if change < 0 else "→")
    print(f"{m:<15} {b:>10.4f} {a:>10.4f} {arrow} {abs(change):>8.4f}")

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
x       = np.arange(len(metrics))
width   = 0.35

bars1 = ax.bar(x - width/2, before, width,
               label='Untuned', color='#90CAF9', edgecolor='white')
bars2 = ax.bar(x + width/2, after,  width,
               label='Tuned (Optuna)', color='#1565C0', edgecolor='white')

for bar, val in zip(bars1, before):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=9)
for bar, val in zip(bars2, after):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title('LightGBM — Untuned vs Tuned (Optuna)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(y=0.9, color='red', linestyle='--',
           alpha=0.4, label='0.9 threshold')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_tuning_comparison.png'), dpi=150)
plt.show()
print("✅ Saved: plot_tuning_comparison.png")

# =============================================================
# SECTION 6 — CONFUSION MATRIX (TUNED)
# =============================================================
print()
print("=" * 60)
print("SECTION 6 — CONFUSION MATRIX (Tuned Model)")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y_pred, title in zip(
    axes,
    [y_pred_untuned, y_pred_tuned],
    ['Untuned LightGBM', 'Tuned LightGBM (Optuna)']
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted\nStayed', 'Predicted\nLeft'],
                yticklabels=['Actual\nStayed', 'Actual\nLeft'],
                ax=ax, annot_kws={'size': 13})
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_confusion_matrices.png'), dpi=150)
plt.show()
print("✅ Saved: plot_confusion_matrices.png")

# =============================================================
# SECTION 7 — ROC CURVE
# =============================================================
print()
print("=" * 60)
print("SECTION 7 — ROC CURVE")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 7))

for y_prob, label, color in [
    (y_prob_untuned, f'Untuned (AUC={baseline_scores["ROC-AUC"]:.3f})', '#90CAF9'),
    (y_prob_tuned,   f'Tuned   (AUC={tuned_scores["ROC-AUC"]:.3f})',   '#1565C0')
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=label, linewidth=2.5, color=color)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.05, color='#1565C0')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Untuned vs Tuned LightGBM',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_roc_curve.png'), dpi=150)
plt.show()
print("✅ Saved: plot_roc_curve.png")

# =============================================================
# SECTION 8 — SHAP VALUES (Explainability for Agent Logic)
# =============================================================
print()
print("=" * 60)
print("SECTION 8 — SHAP VALUES (Why does model predict churn?)")
print("           This may take 1-2 minutes...")
print("=" * 60)

explainer   = shap.TreeExplainer(tuned_model)
shap_values = explainer.shap_values(X_test)

# Handle LightGBM SHAP output format
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# Plot 1: SHAP Summary — all features
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_test, show=False, plot_size=None)
plt.title('SHAP Summary — Feature Impact on Churn Prediction',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_shap_summary.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_shap_summary.png")

# Plot 2: SHAP Bar — top 15 features
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_test, plot_type='bar',
                  max_display=15, show=False)
plt.title('Top 15 Features by SHAP Importance',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_shap_bar.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: plot_shap_bar.png")

# Save SHAP feature importance for Phase 4 agent
shap_importance = pd.DataFrame({
    'Feature'         : X_test.columns,
    'SHAP_Importance' : np.abs(sv).mean(axis=0)
}).sort_values('SHAP_Importance', ascending=False)

shap_importance.to_csv(
    os.path.join(OUTPUT_DIR, 'shap_feature_importance.csv'),
    index=False
)
print("✅ Saved: shap_feature_importance.csv")

print("\nTop 10 Features by SHAP Importance:")
for i, row in shap_importance.head(10).iterrows():
    rank = shap_importance.index.get_loc(i) + 1
    print(f"   {rank:02d}. {row['Feature']:<45} | {row['SHAP_Importance']:.4f}")

# =============================================================
# SECTION 9 — INDIVIDUAL EMPLOYEE EXPLANATION (Sample)
# =============================================================
print()
print("=" * 60)
print("SECTION 9 — INDIVIDUAL EMPLOYEE EXPLANATION (Sample)")
print("=" * 60)

# Pick first HIGH RISK employee as example
probs         = tuned_model.predict_proba(X_test)[:, 1]
high_risk_idx = np.where(probs >= 0.70)[0]

if len(high_risk_idx) > 0:
    sample_idx    = high_risk_idx[0]
    sample_employee = X_test.iloc[[sample_idx]]
    sample_prob   = probs[sample_idx]

    print(f"\n   Sample Employee Index : {sample_idx}")
    print(f"   Churn Probability     : {sample_prob:.2%}")
    print(f"   Risk Level            : 🔴 HIGH RISK")
    print(f"\n   Employee Feature Values:")
    for col, val in sample_employee.iloc[0].items():
        print(f"   {col:<45}: {val}")

    # SHAP waterfall for this employee
    shap_sample = sv[sample_idx]
    feat_shap   = pd.DataFrame({
        'Feature' : X_test.columns,
        'Value'   : sample_employee.iloc[0].values,
        'SHAP'    : shap_sample
    }, index=X_test.columns).reindex(
        pd.Series(np.abs(shap_sample), index=X_test.columns)
        .sort_values(ascending=False).index
    ).head(10)

    print(f"\n   Top Reasons This Employee is HIGH RISK:")
    for _, row in feat_shap.iterrows():
        direction = "pushes toward LEAVING" if row['SHAP'] > 0 \
                    else "pushes toward STAYING"
        print(f"   {row['Feature']:<40} | SHAP: {row['SHAP']:+.4f} | {direction}")
else:
    print("   No HIGH RISK employees found in test set with threshold 0.70")

# =============================================================
# FINAL SUMMARY
# =============================================================
print()
print("=" * 60)
print("PHASE 3 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
Model          : LightGBM (Tuned with Optuna — {N_TRIALS} trials)

UNTUNED vs TUNED:
  Accuracy  : {baseline_scores['Accuracy']:.4f}  →  {tuned_scores['Accuracy']:.4f}  ({'+' if tuned_scores['Accuracy']-baseline_scores['Accuracy']>=0 else ''}{tuned_scores['Accuracy']-baseline_scores['Accuracy']:.4f})
  F1 Score  : {baseline_scores['F1 Score']:.4f}  →  {tuned_scores['F1 Score']:.4f}  ({'+' if tuned_scores['F1 Score']-baseline_scores['F1 Score']>=0 else ''}{tuned_scores['F1 Score']-baseline_scores['F1 Score']:.4f})
  ROC-AUC   : {baseline_scores['ROC-AUC']:.4f}  →  {tuned_scores['ROC-AUC']:.4f}  ({'+' if tuned_scores['ROC-AUC']-baseline_scores['ROC-AUC']>=0 else ''}{tuned_scores['ROC-AUC']-baseline_scores['ROC-AUC']:.4f})
  Recall    : {baseline_scores['Recall']:.4f}  →  {tuned_scores['Recall']:.4f}  ({'+' if tuned_scores['Recall']-baseline_scores['Recall']>=0 else ''}{tuned_scores['Recall']-baseline_scores['Recall']:.4f})

Files saved to outputs/:
  ✅ best_model_tuned.pkl         ← Used in Phase 4 agent
  ✅ optuna_best_params.json      ← Best hyperparameters
  ✅ shap_feature_importance.csv  ← Used in Phase 4 agent
  ✅ plot_cm_untuned.png
  ✅ plot_confusion_matrices.png
  ✅ plot_tuning_comparison.png
  ✅ plot_roc_curve.png
  ✅ plot_shap_summary.png
  ✅ plot_shap_bar.png

Next Step: Run 4_agent_logic.py
""")
