"""
Exploratory Data Analysis - Structured Insights Report Generator
================================================================

Loads the cleaned employee attrition dataset, computes key metrics,
prints a formatted console report, and generates a 6-panel summary
dashboard saved to visualizations/eda_summary_dashboard.png.

Author: Ankit
Date  : May 2026
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr

warnings.filterwarnings('ignore')

# ---- CONSTANTS ----
DATA_PATH = os.path.join(os.path.dirname(__file__), 'employee_attrition_raw.csv')
VIZ_DIR = os.path.join(os.path.dirname(__file__), 'visualizations')
DASHBOARD_PATH = os.path.join(VIZ_DIR, 'eda_summary_dashboard.png')
SEPARATOR = '=' * 65
SUBSEP = '-' * 65


# ---- HELPERS ----
def _header(title: str) -> str:
    """Return a decorated section header."""
    return f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}"


def _subheader(title: str) -> str:
    """Return a lighter sub-section header."""
    return f"\n{SUBSEP}\n  {title}\n{SUBSEP}"


def _bullet(text: str) -> str:
    return f"  * {text}"


# ---- DATA LOADING ----
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    # Ensure binary encoding for attrition (Yes/No -> 1/0)
    df['attrition_flag'] = (df['attrition'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    return df


# ---- METRIC COMPUTATION ----
def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute all key metrics and return them as a dictionary."""
    total = len(df)
    features = df.shape[1]
    attrition_rate = df['attrition_flag'].mean() * 100

    # Basic averages
    avg_salary = df['monthly_salary'].mean()
    avg_satisfaction = df['satisfaction_score'].mean()
    avg_tenure = df['years_at_company'].mean()
    ot_mask = df['overtime'].astype(str).str.strip().str.lower() == 'yes'
    overtime_pct = ot_mask.sum() / total * 100

    # Performance distribution
    perf_dist = df['performance_rating'].value_counts(normalize=True).sort_index() * 100

    # Department attrition
    dept_attrition = (
        df.groupby('department')['attrition_flag']
        .mean()
        .sort_values(ascending=False) * 100
    )
    highest_attrition_dept = dept_attrition.idxmax()
    highest_attrition_rate = dept_attrition.max()

    # Salary comparison
    stayed = df[df['attrition_flag'] == 0]
    left = df[df['attrition_flag'] == 1]
    salary_stayed = stayed['monthly_salary'].mean()
    salary_left = left['monthly_salary'].mean()
    salary_diff = salary_stayed - salary_left

    # Satisfaction comparison
    satisfaction_stayed = stayed['satisfaction_score'].mean()
    satisfaction_left = left['satisfaction_score'].mean()

    # Overtime vs attrition
    ot_flag = df['overtime'].astype(str).str.strip().str.lower() == 'yes'
    ot_attrition = df[ot_flag]['attrition_flag'].mean() * 100
    no_ot_attrition = df[~ot_flag]['attrition_flag'].mean() * 100

    # Tenure comparison
    tenure_stayed = stayed['years_at_company'].mean()
    tenure_left = left['years_at_company'].mean()

    # Point-biserial correlations for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c not in ('attrition_flag', 'employee_id', 'attrition')]
    pb_correlations = {}
    for col in numerical_cols:
        valid = df[[col, 'attrition_flag']].dropna()
        if len(valid) > 2:
            corr, pval = pointbiserialr(valid['attrition_flag'], valid[col])
            pb_correlations[col] = {'correlation': corr, 'p_value': pval}
    # Sort by absolute correlation
    pb_sorted = dict(
        sorted(pb_correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
    )

    # Work-life balance comparison
    wlb_stayed = stayed['work_life_balance'].mean()
    wlb_left = left['work_life_balance'].mean()

    return {
        'total': total,
        'features': features,
        'attrition_rate': attrition_rate,
        'avg_salary': avg_salary,
        'avg_satisfaction': avg_satisfaction,
        'avg_tenure': avg_tenure,
        'overtime_pct': overtime_pct,
        'perf_dist': perf_dist,
        'dept_attrition': dept_attrition,
        'highest_attrition_dept': highest_attrition_dept,
        'highest_attrition_rate': highest_attrition_rate,
        'salary_stayed': salary_stayed,
        'salary_left': salary_left,
        'salary_diff': salary_diff,
        'satisfaction_stayed': satisfaction_stayed,
        'satisfaction_left': satisfaction_left,
        'ot_attrition': ot_attrition,
        'no_ot_attrition': no_ot_attrition,
        'tenure_stayed': tenure_stayed,
        'tenure_left': tenure_left,
        'pb_correlations': pb_sorted,
        'wlb_stayed': wlb_stayed,
        'wlb_left': wlb_left,
    }


# ---- CONSOLE REPORT ----
def print_report(m: dict) -> None:
    """Print a beautiful structured report to the console."""

    print("\n")
    print("+=================================================================+")
    print("|        EXPLORATORY DATA ANALYSIS - INSIGHTS REPORT            |")
    print("|        Employee Attrition Dataset                             |")
    print("+=================================================================+")

    # -- SECTION 1: DATASET OVERVIEW --
    print(_header("[1] DATASET OVERVIEW"))
    print(f"  Total Records        : {m['total']:,}")
    print(f"  Total Features       : {m['features']}")
    print(f"  Overall Attrition    : {m['attrition_rate']:.1f}%")
    print(f"  Dataset Type         : Synthetic HR data (cross-sectional)")

    # -- SECTION 2: KEY METRICS --
    print(_header("[2] KEY METRICS"))
    print(f"  Average Monthly Salary   : Rs.{m['avg_salary']:,.0f}")
    print(f"  Average Satisfaction     : {m['avg_satisfaction']:.2f} / 5.0")
    print(f"  Average Tenure           : {m['avg_tenure']:.1f} years")
    print(f"  Overtime Employees       : {m['overtime_pct']:.1f}%")
    print()
    print("  Performance Rating Distribution:")
    for rating, pct in m['perf_dist'].items():
        bar = '#' * int(pct / 2)
        print(f"    Rating {rating} : {pct:5.1f}%  {bar}")

    # -- SECTION 3: TOP INSIGHTS --
    print(_header("[3] TOP INSIGHTS"))
    insights = [
        f"Department with highest attrition: {m['highest_attrition_dept']} "
        f"({m['highest_attrition_rate']:.1f}%)",
        f"Avg salary of employees who stayed: Rs.{m['salary_stayed']:,.0f}  |  "
        f"left: Rs.{m['salary_left']:,.0f}  (diff= Rs.{m['salary_diff']:,.0f})",
        f"Avg satisfaction - Stayed: {m['satisfaction_stayed']:.2f}  |  "
        f"Left: {m['satisfaction_left']:.2f}",
        f"Overtime attrition rate: {m['ot_attrition']:.1f}%  vs  "
        f"Non-overtime: {m['no_ot_attrition']:.1f}%",
        f"Avg tenure - Stayed: {m['tenure_stayed']:.1f} yrs  |  "
        f"Left: {m['tenure_left']:.1f} yrs",
        f"Work-life balance - Stayed: {m['wlb_stayed']:.2f}  |  "
        f"Left: {m['wlb_left']:.2f}",
    ]
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")

    # -- SECTION 4: TOP INFLUENCING FACTORS --
    print(_header("[4] TOP INFLUENCING FACTORS (Point-Biserial Correlation)"))
    print(f"  {'Feature':<25} {'Correlation':>12}  {'p-value':>10}  Significance")
    print(f"  {'-'*25} {'-'*12}  {'-'*10}  {'-'*14}")
    for col, vals in m['pb_correlations'].items():
        sig = '***' if vals['p_value'] < 0.001 else ('**' if vals['p_value'] < 0.01 else ('*' if vals['p_value'] < 0.05 else 'ns'))
        print(f"  {col:<25} {vals['correlation']:>+12.4f}  {vals['p_value']:>10.2e}  {sig}")

    # -- SECTION 5: OBSERVATIONS --
    print(_header("[5] OBSERVATIONS"))
    observations = [
        "Employees who leave tend to have lower monthly salaries compared to those who stay.",
        "Overtime is a strong differentiator - employees working overtime show markedly higher attrition.",
        "Satisfaction scores are noticeably lower among employees who eventually leave the organization.",
        "Shorter-tenured employees are more vulnerable to attrition, suggesting early-career retention gaps.",
        "Work-life balance scores differ between stayed and left groups, indicating its role in turnover.",
        "Department-level attrition varies significantly, pointing to localized management or culture issues.",
    ]
    for obs in observations:
        print(_bullet(obs))

    # -- SECTION 6: RECOMMENDATIONS --
    print(_header("[6] RECOMMENDATIONS"))
    recommendations = [
        "Conduct targeted salary reviews in high-attrition departments to close compensation gaps.",
        "Implement overtime monitoring and cap policies; offer compensatory time-off or incentives.",
        "Launch quarterly employee satisfaction pulse surveys with anonymous feedback channels.",
        "Design structured onboarding and mentorship programs for employees in their first 2 years.",
        "Empower department heads with retention dashboards to track leading indicators in real time.",
    ]
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print(f"\n{SEPARATOR}")
    print("  Report generated successfully.")
    print(SEPARATOR)


# ---- DASHBOARD ----
def generate_dashboard(df: pd.DataFrame, m: dict) -> None:
    """Create a 6-panel summary dashboard and save as PNG."""
    os.makedirs(VIZ_DIR, exist_ok=True)

    sns.set_style('whitegrid')
    palette_main = ['#2ecc71', '#e74c3c']
    palette_dept = sns.color_palette('coolwarm', n_colors=len(m['dept_attrition']))

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='#f8f9fa')
    fig.suptitle('Employee Attrition - EDA Summary Dashboard',
                 fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')

    # -- Panel 1: Attrition Pie Chart --
    ax1 = axes[0, 0]
    attrition_counts = df['attrition_flag'].value_counts()
    labels = ['Stayed', 'Left']
    ax1.pie(attrition_counts, labels=labels, autopct='%1.1f%%',
            colors=palette_main, startangle=90,
            explode=(0, 0.06), textprops={'fontsize': 11},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax1.set_title('Attrition Distribution', fontsize=13, fontweight='bold', pad=12)

    # -- Panel 2: Department Attrition Rate Bar --
    ax2 = axes[0, 1]
    dept_data = m['dept_attrition'].sort_values(ascending=True)
    bars = ax2.barh(dept_data.index, dept_data.values, color=palette_dept, edgecolor='white')
    ax2.set_xlabel('Attrition Rate (%)', fontsize=11)
    ax2.set_title('Attrition Rate by Department', fontsize=13, fontweight='bold', pad=12)
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}%', va='center', fontsize=9, color='#34495e')

    # -- Panel 3: Salary Comparison (Stayed vs Left) --
    ax3 = axes[0, 2]
    stayed = df[df['attrition_flag'] == 0]['monthly_salary']
    left = df[df['attrition_flag'] == 1]['monthly_salary']
    bp = ax3.boxplot([stayed, left], labels=['Stayed', 'Left'],
                     patch_artist=True, widths=0.5,
                     boxprops=dict(linewidth=1.2),
                     medianprops=dict(color='#2c3e50', linewidth=2))
    bp['boxes'][0].set_facecolor(palette_main[0])
    bp['boxes'][1].set_facecolor(palette_main[1])
    ax3.set_ylabel('Monthly Salary (Rs.)', fontsize=11)
    ax3.set_title('Salary: Stayed vs Left', fontsize=13, fontweight='bold', pad=12)

    # -- Panel 4: Satisfaction Distribution by Attrition --
    ax4 = axes[1, 0]
    sns.histplot(data=df, x='satisfaction_score', hue='attrition_flag',
                 kde=True, palette=palette_main, ax=ax4,
                 element='step', stat='density', common_norm=False, alpha=0.35)
    ax4.set_xlabel('Satisfaction Score', fontsize=11)
    ax4.set_title('Satisfaction Distribution by Attrition', fontsize=13,
                  fontweight='bold', pad=12)
    handles = ax4.get_legend().legend_handles
    ax4.legend(handles, ['Stayed', 'Left'], title='Attrition')

    # -- Panel 5: Overtime vs Attrition --
    ax5 = axes[1, 1]
    ot_map = df.groupby('overtime')['attrition_flag'].mean() * 100
    ot_labels = ot_map.index.tolist()
    ot_vals = ot_map.values
    bar_colors = ['#2ecc71', '#e74c3c'] if len(ot_labels) == 2 else palette_main
    ot_bars = ax5.bar(ot_labels, ot_vals, color=bar_colors[:len(ot_labels)],
                      edgecolor='white', width=0.5)
    ax5.set_ylabel('Attrition Rate (%)', fontsize=11)
    ax5.set_title('Overtime vs Attrition Rate', fontsize=13, fontweight='bold', pad=12)
    for bar in ot_bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 f'{height:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#34495e')

    # -- Panel 6: Top Influencing Factors --
    ax6 = axes[1, 2]
    top_factors = dict(list(m['pb_correlations'].items())[:8])
    factor_names = list(top_factors.keys())
    factor_vals = [v['correlation'] for v in top_factors.values()]
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in factor_vals]
    ax6.barh(factor_names[::-1], factor_vals[::-1], color=colors[::-1], edgecolor='white')
    ax6.set_xlabel('Point-Biserial Correlation', fontsize=11)
    ax6.set_title('Top Influencing Factors', fontsize=13, fontweight='bold', pad=12)
    ax6.axvline(x=0, color='#7f8c8d', linewidth=0.8, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DASHBOARD_PATH, dpi=200, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close(fig)
    print(f"\n  [OK] Dashboard saved -> {DASHBOARD_PATH}")


# ---- MAIN ----
def main():
    print("  Loading dataset ...")
    df = load_data()

    print("  Computing metrics ...")
    metrics = compute_metrics(df)

    print_report(metrics)
    generate_dashboard(df, metrics)


if __name__ == '__main__':
    main()
