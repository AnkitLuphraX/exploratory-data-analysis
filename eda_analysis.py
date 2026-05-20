"""
Employee Attrition - Exploratory Data Analysis (EDA)
=====================================================
A comprehensive EDA pipeline that produces statistical summaries,
12 publication-quality visualizations, and formal hypothesis tests
to identify the key drivers of employee attrition.

Author : Ankit
Date   : May 2026
Dataset: employee_attrition_raw.csv
"""

# ============================================================
# 0. IMPORTS & CONFIGURATION
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency

# Plotting defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
})

PALETTE_ATTRITION = {'Yes': '#e74c3c', 'No': '#3498db'}
VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 70)
print("  EMPLOYEE ATTRITION - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'employee_attrition_raw.csv')

if not os.path.exists(DATA_PATH):
    sys.exit(f"[ERROR] Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset loaded from: {DATA_PATH}")


# ############################################################
# PART A : STATISTICAL SUMMARIES
# ############################################################
print("\n" + "=" * 70)
print("  PART A : STATISTICAL SUMMARIES")
print("=" * 70)

# ---- A1. Shape, dtypes, memory ----
print("\n--- A1. Dataset Overview ---")
print(f"  Rows   : {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print("\nColumn Data Types:")
print(df.dtypes.to_string())

# ---- A2. Descriptive statistics ----
print("\n--- A2. Descriptive Statistics (Numerical Columns) ---")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove employee_id if present
if 'employee_id' in num_cols:
    num_cols.remove('employee_id')

desc = df[num_cols].describe().T
desc['median'] = df[num_cols].median()
desc['skewness'] = df[num_cols].skew()
desc['kurtosis'] = df[num_cols].kurtosis()
desc = desc[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']]
print(desc.round(3).to_string())

# ---- A3. Missing values ----
print("\n--- A3. Missing Value Check ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Pct (%)': missing_pct})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
if missing_df.empty:
    print("  [OK] No missing values detected.")
else:
    print(missing_df.to_string())

# ---- A4. Categorical value counts ----
print("\n--- A4. Value Counts for Categorical Columns ---")
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    print(f"\n  [{col}]")
    vc = df[col].value_counts()
    for val, cnt in vc.items():
        print(f"    {val:.<30s} {cnt:>6,}  ({cnt / len(df) * 100:.1f}%)")

# ---- A5. Overall attrition rate ----
print("\n--- A5. Overall Attrition Rate ---")
attr_counts = df['attrition'].value_counts()
attr_rate = (attr_counts.get('Yes', 0) / len(df)) * 100
print(f"  Attrition = Yes : {attr_counts.get('Yes', 0):>6,}  ({attr_rate:.2f}%)")
print(f"  Attrition = No  : {attr_counts.get('No', 0):>6,}  ({100 - attr_rate:.2f}%)")

# ---- A6. Grouped summary by attrition ----
print("\n--- A6. Summary Statistics Grouped by Attrition ---")
grouped = df.groupby('attrition')[num_cols].agg(['mean', 'median', 'std'])
print(grouped.round(3).to_string())


# ############################################################
# PART B : VISUALIZATIONS  (12 Charts)
# ############################################################
print("\n" + "=" * 70)
print("  PART B : VISUALIZATIONS")
print("=" * 70)


def save_fig(fig, fname):
    """Save figure and print confirmation."""
    path = os.path.join(VIZ_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Saved: {fname}")


# ---- B1. Attrition Overview ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Attrition Overview')

# Pie chart
counts = df['attrition'].value_counts()
colors = [PALETTE_ATTRITION[k] for k in counts.index]
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
            colors=colors, explode=[0.04] * len(counts), textprops={'fontsize': 12})
axes[0].set_title('Attrition Distribution (%)')

# Bar chart with value labels
bars = axes[1].bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.5)
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')
axes[1].set_title('Attrition Count')
axes[1].set_ylabel('Number of Employees')
save_fig(fig, '01_attrition_overview.png')


# ---- B2. Salary Distribution ----
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle('Monthly Salary Distribution by Attrition')
for label, color in PALETTE_ATTRITION.items():
    subset = df[df['attrition'] == label]['monthly_salary']
    ax.hist(subset, bins=40, alpha=0.45, label=f'Attrition={label}', color=color, edgecolor='white')
    sns.kdeplot(subset, ax=ax, color=color, linewidth=2)
    ax.axvline(subset.mean(), color=color, linestyle='--', linewidth=1.5,
               label=f'Mean ({label}): ${subset.mean():,.0f}')
ax.set_xlabel('Monthly Salary ($)')
ax.set_ylabel('Frequency')
ax.legend(frameon=True)
save_fig(fig, '02_salary_distribution.png')


# ---- B3. Department Analysis ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Department-wise Analysis')

dept_attr = df.groupby('department')['attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
bars = axes[0].bar(dept_attr.index, dept_attr.values, color='#e67e22', edgecolor='white')
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
axes[0].set_title('Attrition Rate by Department')
axes[0].set_ylabel('Attrition Rate (%)')
axes[0].tick_params(axis='x', rotation=30)

dept_sal = df.groupby(['department', 'attrition'])['monthly_salary'].mean().unstack()
dept_sal.plot(kind='bar', ax=axes[1], color=[PALETTE_ATTRITION['No'], PALETTE_ATTRITION['Yes']],
              edgecolor='white')
axes[1].set_title('Avg Salary by Department & Attrition')
axes[1].set_ylabel('Average Salary ($)')
axes[1].legend(title='Attrition', frameon=True)
axes[1].tick_params(axis='x', rotation=30)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig, '03_department_analysis.png')


# ---- B4. Correlation Heatmap ----
fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle('Correlation Heatmap (Numerical Features)')
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('')
save_fig(fig, '04_correlation_heatmap.png')


# ---- B5. Age vs Satisfaction ----
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Age vs Satisfaction Score by Attrition')
for label, color in PALETTE_ATTRITION.items():
    subset = df[df['attrition'] == label]
    ax.scatter(subset['age'], subset['satisfaction_score'], alpha=0.35, s=25,
               c=color, label=f'Attrition={label}', edgecolors='none')
    # Trend line
    z = np.polyfit(subset['age'], subset['satisfaction_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset['age'].min(), subset['age'].max(), 100)
    ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle='--')
ax.set_xlabel('Age')
ax.set_ylabel('Satisfaction Score')
ax.legend(frameon=True)
save_fig(fig, '05_age_vs_satisfaction.png')


# ---- B6. Tenure Analysis ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Tenure Analysis: Years at Company by Attrition')

sns.violinplot(data=df, x='attrition', y='years_at_company', palette=PALETTE_ATTRITION,
               ax=axes[0], inner='quartile')
axes[0].set_title('Violin Plot')
axes[0].set_ylabel('Years at Company')

sns.boxplot(data=df, x='attrition', y='years_at_company', palette=PALETTE_ATTRITION,
            ax=axes[1], width=0.4)
axes[1].set_title('Box Plot')
axes[1].set_ylabel('Years at Company')
save_fig(fig, '06_tenure_analysis.png')


# ---- B7. Overtime Impact ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Overtime Impact on Attrition')

ot_ct = df.groupby(['overtime', 'attrition']).size().unstack(fill_value=0)
ot_ct.plot(kind='bar', ax=axes[0], color=[PALETTE_ATTRITION['No'], PALETTE_ATTRITION['Yes']],
           edgecolor='white')
axes[0].set_title('Count by Overtime & Attrition')
axes[0].set_ylabel('Number of Employees')
axes[0].legend(title='Attrition', frameon=True)
axes[0].tick_params(axis='x', rotation=0)

ot_rate = df.groupby('overtime')['attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
bars = axes[1].bar(ot_rate.index, ot_rate.values, color=['#3498db', '#e74c3c'], edgecolor='white', width=0.45)
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')
axes[1].set_title('Attrition Rate by Overtime')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].tick_params(axis='x', rotation=0)
save_fig(fig, '07_overtime_impact.png')


# ---- B8. Education Analysis ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Education Level Analysis')

edu_rate = df.groupby('education_level')['attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
bars = axes[0].bar(edu_rate.index, edu_rate.values, color='#9b59b6', edgecolor='white')
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
axes[0].set_title('Attrition Rate by Education Level')
axes[0].set_ylabel('Attrition Rate (%)')
axes[0].tick_params(axis='x', rotation=25)

edu_dist = df['education_level'].value_counts()
axes[1].pie(edu_dist, labels=edu_dist.index, autopct='%1.1f%%', startangle=140,
            textprops={'fontsize': 10})
axes[1].set_title('Education Distribution')
fig.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig, '08_education_analysis.png')


# ---- B9. Performance vs Salary ----
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Monthly Salary by Performance Rating')
palette_perf = sns.color_palette('viridis', n_colors=df['performance_rating'].nunique())
sns.boxplot(data=df, x='performance_rating', y='monthly_salary', palette=palette_perf,
            ax=ax, width=0.5)
ax.set_xlabel('Performance Rating')
ax.set_ylabel('Monthly Salary ($)')
save_fig(fig, '09_performance_salary.png')


# ---- B10. Work-Life Balance ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Work-Life Balance Impact')

wlb_rate = df.groupby('work_life_balance')['attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
wlb_ct = df.groupby(['work_life_balance', 'attrition']).size().unstack(fill_value=0)
wlb_ct.plot(kind='bar', ax=axes[0], color=[PALETTE_ATTRITION['No'], PALETTE_ATTRITION['Yes']],
            edgecolor='white')
axes[0].set_title('Work-Life Balance vs Attrition Count')
axes[0].set_ylabel('Count')
axes[0].legend(title='Attrition', frameon=True)
axes[0].tick_params(axis='x', rotation=0)

wlb_sat = df.groupby('work_life_balance')['satisfaction_score'].mean()
ax2 = axes[1]
bars = ax2.bar(wlb_sat.index.astype(str), wlb_sat.values, color='#1abc9c', edgecolor='white', width=0.45)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
ax2.set_title('Avg Satisfaction by Work-Life Balance')
ax2.set_xlabel('Work-Life Balance Score')
ax2.set_ylabel('Avg Satisfaction Score')
save_fig(fig, '10_work_life_balance.png')


# ---- B11. Pairplot ----
print("  >>> Generating pairplot (this may take a moment)...")
pair_cols = ['monthly_salary', 'satisfaction_score', 'years_at_company', 'avg_weekly_hours', 'attrition']
g = sns.pairplot(df[pair_cols], hue='attrition', palette=PALETTE_ATTRITION,
                 diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15, 'edgecolor': 'none'})
g.figure.suptitle('Pairplot of Key Numerical Features', y=1.02)
save_fig(g.figure, '11_pairplot_features.png')


# ---- B12. Key Factors Summary ----
df_binary = df.copy()
df_binary['attrition_bin'] = (df_binary['attrition'] == 'Yes').astype(int)

pb_results = {}
for col in num_cols:
    corr_val, p_val = pointbiserialr(df_binary['attrition_bin'], df_binary[col])
    pb_results[col] = corr_val

pb_series = pd.Series(pb_results).sort_values(key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle('Key Factors Influencing Attrition\n(Point-Biserial Correlation with Attrition)')
colors = ['#e74c3c' if v > 0 else '#3498db' for v in pb_series.values]
bars = ax.barh(pb_series.index, pb_series.values, color=colors, edgecolor='white', height=0.6)
ax.axvline(0, color='grey', linewidth=0.8, linestyle='-')
ax.set_xlabel('Point-Biserial Correlation')
ax.set_ylabel('')
for bar, val in zip(bars, pb_series.values):
    offset = 0.005 if val >= 0 else -0.005
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
# Legend patch
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Positive (increases attrition)'),
                   Patch(facecolor='#3498db', label='Negative (decreases attrition)')]
ax.legend(handles=legend_elements, loc='lower right', frameon=True)
save_fig(fig, '12_key_factors.png')

print(f"\n  [OK] All 12 visualizations saved to: {VIZ_DIR}")


# ############################################################
# PART C : STATISTICAL TESTS
# ############################################################
print("\n" + "=" * 70)
print("  PART C : STATISTICAL TESTS")
print("=" * 70)

# ---- C1. Pearson Correlation Matrix ----
print("\n--- C1. Pearson Correlation Matrix (Numerical Features) ---")
pearson_corr = df[num_cols].corr(method='pearson')
print(pearson_corr.round(3).to_string())

# ---- C2. Point-Biserial Correlations ----
print("\n--- C2. Point-Biserial Correlation: Numerical Features vs Attrition ---")
print(f"  {'Feature':<25s}  {'Correlation':>12s}  {'p-value':>12s}  {'Significant (alpha=0.05)':>22s}")
print("  " + "-" * 75)
pb_full = {}
for col in num_cols:
    corr_val, p_val = pointbiserialr(df_binary['attrition_bin'], df_binary[col])
    sig = "Yes ***" if p_val < 0.001 else ("Yes **" if p_val < 0.01 else ("Yes *" if p_val < 0.05 else "No"))
    print(f"  {col:<25s}  {corr_val:>12.4f}  {p_val:>12.4e}  {sig:>22s}")
    pb_full[col] = {'correlation': corr_val, 'p_value': p_val}

# ---- C3. Chi-Square Tests ----
print("\n--- C3. Chi-Square Tests of Independence (vs Attrition) ---")
chi_sq_vars = ['overtime', 'department', 'education_level', 'gender']
print(f"  {'Variable':<20s}  {'Chi2':>10s}  {'p-value':>12s}  {'DoF':>5s}  {'Significant (alpha=0.05)':>22s}")
print("  " + "-" * 75)
for var in chi_sq_vars:
    ct = pd.crosstab(df[var], df['attrition'])
    chi2, p, dof, expected = chi2_contingency(ct)
    sig = "Yes ***" if p < 0.001 else ("Yes **" if p < 0.01 else ("Yes *" if p < 0.05 else "No"))
    print(f"  {var:<20s}  {chi2:>10.2f}  {p:>12.4e}  {dof:>5d}  {sig:>22s}")

# ---- C4. Ranked Key Factors ----
print("\n--- C4. Ranked Key Factors Influencing Attrition ---")
print("  (Sorted by absolute point-biserial correlation)")
ranked = sorted(pb_full.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
print(f"\n  {'Rank':<6s}  {'Feature':<25s}  {'|r|':>8s}  {'Direction':>10s}")
print("  " + "-" * 55)
for i, (feat, vals) in enumerate(ranked, 1):
    direction = "Positive" if vals['correlation'] > 0 else "Negative"
    print(f"  {i:<6d}  {feat:<25s}  {abs(vals['correlation']):>8.4f}  {direction:>10s}")


# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 70)
print("  [OK]  EDA ANALYSIS COMPLETE")
print("=" * 70)
print(f"  - Console summaries  : Parts A & C above")
print(f"  - Visualizations (12): {VIZ_DIR}")
print("=" * 70)
