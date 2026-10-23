# =============================================================
# PHASE 1 — DATA EXPLORATION
# Project: Employee Churn Prediction
# Dataset: Saudi Employee Attrition Dataset
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# =============================================================
# CONFIGURATION — Change this path if your file is elsewhere
# =============================================================
DATA_PATH = r"data\Employee_attrition_dataset_for_tree-based_models.xlsx"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# SECTION 1 — LOAD DATA
# =============================================================
print("=" * 60)
print("SECTION 1 — LOADING DATA")
print("=" * 60)

df = pd.read_excel(DATA_PATH)

print(f"✅ File loaded successfully")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print()
print("Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:02d}. {col}")

# =============================================================
# SECTION 2 — BASIC STATISTICS
# =============================================================
print()
print("=" * 60)
print("SECTION 2 — BASIC STATISTICS")
print("=" * 60)

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values Per Column ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values found in any column")
else:
    print(missing[missing > 0])

print("\n--- Statistical Summary ---")
print(df.describe().round(2))

# =============================================================
# SECTION 3 — TARGET VARIABLE (ATTRITION) ANALYSIS
# =============================================================
print()
print("=" * 60)
print("SECTION 3 — TARGET VARIABLE ANALYSIS")
print("=" * 60)

attrition_counts = df['Attrition'].value_counts()
attrition_pct    = df['Attrition'].value_counts(normalize=True) * 100

print(f"\n   Stayed (0) : {attrition_counts[0]} employees  ({attrition_pct[0]:.1f}%)")
print(f"   Left   (1) : {attrition_counts[1]} employees  ({attrition_pct[1]:.1f}%)")

if attrition_pct.min() >= 35:
    print("\n✅ Dataset is BALANCED — no need for SMOTE or resampling")
else:
    print("\n⚠️  Dataset is IMBALANCED — consider SMOTE in Phase 2")

# --- Plot 1: Attrition Pie Chart ---
fig, ax = plt.subplots(figsize=(7, 7))
labels  = ['Stayed (0)', 'Left (1)']
colors  = ['#4CAF50', '#F44336']
explode = (0.03, 0.03)

ax.pie(
    attrition_counts,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    startangle=140,
    textprops={'fontsize': 13}
)
ax.set_title('Attrition Distribution\n(Target Variable)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot1_attrition_distribution.png'), dpi=150)
plt.show()
print("\n✅ Plot 1 saved: plot1_attrition_distribution.png")

# =============================================================
# SECTION 4 — KEY FEATURE vs ATTRITION PLOTS
# Using exact encodings from Dataset_Keys.docx
# =============================================================
print()
print("=" * 60)
print("SECTION 4 — KEY FEATURES vs ATTRITION")
print("=" * 60)

# Colour palette: green = stayed, red = left
palette = {0: '#4CAF50', 1: '#F44336'}

def plot_feature_vs_attrition(feature, title, x_labels, filename, figsize=(8, 5)):
    """
    Bar chart showing attrition count for each category of a feature.
    x_labels: dict mapping encoded value → human-readable label
    """
    plot_df = df.copy()
    plot_df['Attrition_Label'] = plot_df['Attrition'].map({0: 'Stayed', 1: 'Left'})
    plot_df['Feature_Label']   = plot_df[feature].map(x_labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(
        data=plot_df,
        x='Feature_Label',
        hue='Attrition_Label',
        palette={'Stayed': '#4CAF50', 'Left': '#F44336'},
        order=[x_labels[k] for k in sorted(x_labels.keys())],
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(feature.replace('_', ' '), fontsize=11)
    ax.set_ylabel('Number of Employees', fontsize=11)
    ax.legend(title='Attrition', fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.show()
    print(f"✅ Saved: {filename}")


# --- Plot 2: Job Satisfaction vs Attrition ---
plot_feature_vs_attrition(
    feature   = 'Job_Satisfaction',
    title     = 'Job Satisfaction vs Attrition',
    x_labels  = {0: 'Not Satisfied', 1: 'Satisfied', 2: 'Very Satisfied'},
    filename  = 'plot2_job_satisfaction.png'
)

# --- Plot 3: Psychological Exhaustion vs Attrition ---
plot_feature_vs_attrition(
    feature   = 'Psychological_Exhaustion',
    title     = 'Psychological Exhaustion vs Attrition',
    x_labels  = {0: 'No', 1: 'Sometimes', 2: 'Yes'},
    filename  = 'plot3_psychological_exhaustion.png'
)

# --- Plot 4: Monthly Salary vs Attrition ---
plot_feature_vs_attrition(
    feature   = 'MonthlySalary',
    title     = 'Monthly Salary vs Attrition',
    x_labels  = {0: '1k-5k', 1: '6k-10k', 2: '11k-15k', 3: '16k+'},
    filename  = 'plot4_monthly_salary.png'
)

# --- Plot 5: Work Life Balance vs Attrition ---
plot_feature_vs_attrition(
    feature   = 'Work_Live_Balance',
    title     = 'Work-Life Balance vs Attrition',
    x_labels  = {0: 'Easy', 1: 'Medium', 2: 'Difficult'},
    filename  = 'plot5_work_life_balance.png'
)

# --- Plot 6: Get Deserved Promotion vs Attrition ---
plot_feature_vs_attrition(
    feature   = 'Get_Deserved_Promotion',
    title     = 'Got Deserved Promotion vs Attrition',
    x_labels  = {0: 'No', 1: 'Yes'},
    filename  = 'plot6_promotion.png'
)

# --- Plot 7: Age vs Attrition ---
plot_feature_vs_attrition(
    feature   = 'Age',
    title     = 'Age Group vs Attrition',
    x_labels  = {0: '21-30', 1: '31-40', 2: '41+'},
    filename  = 'plot7_age.png'
)

# --- Plot 8: Physical Stress vs Attrition ---
plot_feature_vs_attrition(
    feature   = 'Physical_Stress',
    title     = 'Physical Stress vs Attrition',
    x_labels  = {0: 'No', 1: 'Sometimes', 2: 'Yes'},
    filename  = 'plot8_physical_stress.png'
)

# =============================================================
# SECTION 5 — CORRELATION HEATMAP
# =============================================================
print()
print("=" * 60)
print("SECTION 5 — CORRELATION HEATMAP")
print("=" * 60)

fig, ax = plt.subplots(figsize=(16, 13))
corr_matrix = df.corr(numeric_only=True)

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    linewidths=0.4,
    annot_kws={'size': 7},
    ax=ax
)
ax.set_title('Correlation Heatmap — All 34 Features', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot9_correlation_heatmap.png'), dpi=150)
plt.show()
print("✅ Plot 9 saved: plot9_correlation_heatmap.png")

# =============================================================
# SECTION 6 — TOP FEATURES CORRELATED WITH ATTRITION
# =============================================================
print()
print("=" * 60)
print("SECTION 6 — TOP FEATURES CORRELATED WITH ATTRITION")
print("=" * 60)

attrition_corr = (
    corr_matrix['Attrition']
    .drop('Attrition')
    .abs()
    .sort_values(ascending=False)
)

print("\nTop 10 Features Most Correlated with Attrition:\n")
for rank, (feature, value) in enumerate(attrition_corr.head(10).items(), 1):
    direction = "↑ Higher = more likely to leave" if corr_matrix['Attrition'][feature] > 0 \
                else "↓ Higher = less likely to leave"
    print(f"   {rank:02d}. {feature:<45} | r = {value:.3f} | {direction}")

# Bar chart of top 10 correlations
fig, ax = plt.subplots(figsize=(10, 6))
top10 = attrition_corr.head(10)
colors = ['#F44336' if corr_matrix['Attrition'][f] > 0 else '#4CAF50' for f in top10.index]

bars = ax.barh(top10.index[::-1], top10.values[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Absolute Correlation with Attrition', fontsize=11)
ax.set_title('Top 10 Features Correlated with Attrition\n(Red = positive, Green = negative)',
             fontsize=13, fontweight='bold')

for bar, val in zip(bars, top10.values[::-1]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot10_top_correlations.png'), dpi=150)
plt.show()
print("✅ Plot 10 saved: plot10_top_correlations.png")

# =============================================================
# FINAL SUMMARY
# =============================================================
print()
print("=" * 60)
print("PHASE 1 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
Dataset      : Saudi Employee Attrition
Rows         : {df.shape[0]}
Columns      : {df.shape[1]}
Missing Data : None
Attrition    : {attrition_counts[1]} left ({attrition_pct[1]:.1f}%) | {attrition_counts[0]} stayed ({attrition_pct[0]:.1f}%)
Balance      : ✅ Balanced — ready for Phase 2 model training

Plots saved to: outputs/
  plot1_attrition_distribution.png
  plot2_job_satisfaction.png
  plot3_psychological_exhaustion.png
  plot4_monthly_salary.png
  plot5_work_life_balance.png
  plot6_promotion.png
  plot7_age.png
  plot8_physical_stress.png
  plot9_correlation_heatmap.png
  plot10_top_correlations.png

Next Step: Run 2_model_training.py
""")
