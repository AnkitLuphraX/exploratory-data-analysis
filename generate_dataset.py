"""
generate_dataset.py - Synthetic Employee Attrition / HR Analytics Dataset Generator
====================================================================================

Generates a realistic synthetic dataset of ~2,000 employee records with correlated
features suitable for Exploratory Data Analysis (EDA) on employee attrition.

Key design choices:
  * Attrition is NOT random - it is driven by satisfaction, overtime, salary,
    distance-from-home, and work-life balance so that downstream EDA reveals
    meaningful patterns.
  * Salary is correlated with education level, job role seniority, and tenure.
  * Overtime probability rises with weekly hours; satisfaction drops with overtime.

Author : Ankit
Date   : May 2026
"""

import numpy as np
import pandas as pd

# -----------------------------------------------
# 1. Configuration & Seed
# -----------------------------------------------
SEED = 42
NUM_RECORDS = 2000
OUTPUT_FILE = "employee_attrition_raw.csv"

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# -----------------------------------------------
# 2. Employee ID
# -----------------------------------------------
employee_ids = [f"EMP-{i}" for i in range(1001, 1001 + NUM_RECORDS)]

# -----------------------------------------------
# 3. Demographics
# -----------------------------------------------
# Age -- normal distribution centred at 36, clipped to 22-60
ages = np.clip(np.round(rng.normal(loc=36, scale=8, size=NUM_RECORDS)).astype(int), 22, 60)

# Gender
genders = rng.choice(["Male", "Female", "Other"], size=NUM_RECORDS, p=[0.50, 0.40, 0.10])

# Marital status
marital_statuses = rng.choice(
    ["Single", "Married", "Divorced"], size=NUM_RECORDS, p=[0.30, 0.55, 0.15]
)

# -----------------------------------------------
# 4. Department & Job Role
# -----------------------------------------------
DEPARTMENTS = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]
DEPT_WEIGHTS = [0.30, 0.20, 0.10, 0.15, 0.10, 0.15]

departments = rng.choice(DEPARTMENTS, size=NUM_RECORDS, p=DEPT_WEIGHTS)

# Role weights vary by department for realism
ROLE_MAP = {
    "Engineering":  {"Senior Engineer": 0.30, "Analyst": 0.20, "Manager": 0.15,
                     "Team Lead": 0.15, "Associate": 0.10, "Director": 0.05, "Executive": 0.05},
    "Sales":        {"Associate": 0.30, "Analyst": 0.20, "Manager": 0.15,
                     "Team Lead": 0.15, "Executive": 0.10, "Senior Engineer": 0.05, "Director": 0.05},
    "HR":           {"Associate": 0.25, "Analyst": 0.25, "Manager": 0.20,
                     "Team Lead": 0.10, "Director": 0.10, "Executive": 0.05, "Senior Engineer": 0.05},
    "Marketing":    {"Analyst": 0.25, "Associate": 0.25, "Manager": 0.15,
                     "Team Lead": 0.15, "Executive": 0.10, "Director": 0.05, "Senior Engineer": 0.05},
    "Finance":      {"Analyst": 0.30, "Manager": 0.20, "Associate": 0.20,
                     "Team Lead": 0.10, "Senior Engineer": 0.05, "Director": 0.10, "Executive": 0.05},
    "Operations":   {"Associate": 0.25, "Analyst": 0.20, "Team Lead": 0.20,
                     "Manager": 0.15, "Senior Engineer": 0.05, "Director": 0.10, "Executive": 0.05},
}

job_roles = []
for dept in departments:
    roles = list(ROLE_MAP[dept].keys())
    probs = list(ROLE_MAP[dept].values())
    job_roles.append(rng.choice(roles, p=probs))
job_roles = np.array(job_roles)

# -----------------------------------------------
# 5. Education Level
# -----------------------------------------------
EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]
EDUCATION_WEIGHTS = [0.10, 0.45, 0.35, 0.10]

education_levels = rng.choice(EDUCATION_LEVELS, size=NUM_RECORDS, p=EDUCATION_WEIGHTS)

# Numeric mapping for salary correlation
EDUCATION_RANK = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
education_rank = np.array([EDUCATION_RANK[e] for e in education_levels])

# -----------------------------------------------
# 6. Tenure & Promotion
# -----------------------------------------------
# Years at company -- exponential-ish, clipped 0-30
years_at_company = np.clip(
    np.round(rng.exponential(scale=5, size=NUM_RECORDS)).astype(int), 0, 30
)

# Years since last promotion -- 0 to min(10, tenure)
years_since_promotion = np.array([
    rng.integers(0, min(10, yac) + 1) for yac in years_at_company
])

# -----------------------------------------------
# 7. Projects
# -----------------------------------------------
num_projects = rng.integers(1, 11, size=NUM_RECORDS)

# -----------------------------------------------
# 8. Working Hours & Overtime
# -----------------------------------------------
avg_weekly_hours = np.round(rng.uniform(35, 60, size=NUM_RECORDS), 1)

# Overtime -- higher probability when hours > 45
overtime_probs = np.where(avg_weekly_hours > 50, 0.80,
                 np.where(avg_weekly_hours > 45, 0.55, 0.15))
overtime = np.array(["Yes" if rng.random() < p else "No" for p in overtime_probs])

# -----------------------------------------------
# 9. Satisfaction & Work-Life Balance
# -----------------------------------------------
# Satisfaction inversely correlated with overtime
base_satisfaction = rng.uniform(1.0, 5.0, size=NUM_RECORDS)
satisfaction_penalty = np.where(overtime == "Yes", rng.uniform(0.5, 1.5, size=NUM_RECORDS), 0)
satisfaction_score = np.round(np.clip(base_satisfaction - satisfaction_penalty, 1.0, 5.0), 1)

# Work-life balance inversely correlated with overtime & high hours
base_wlb = rng.integers(1, 6, size=NUM_RECORDS)
wlb_penalty = (overtime == "Yes").astype(int) + (avg_weekly_hours > 50).astype(int)
work_life_balance = np.clip(base_wlb - wlb_penalty, 1, 5)

# -----------------------------------------------
# 10. Performance Rating
# -----------------------------------------------
performance_rating = rng.choice([1, 2, 3, 4, 5], size=NUM_RECORDS,
                                p=[0.05, 0.10, 0.35, 0.35, 0.15])

# -----------------------------------------------
# 11. Monthly Salary (Rs.)
# -----------------------------------------------
ROLE_SALARY_BASE = {
    "Associate": 30000, "Analyst": 40000, "Team Lead": 55000,
    "Senior Engineer": 65000, "Manager": 75000, "Director": 100000,
    "Executive": 120000,
}

monthly_salary = []
for i in range(NUM_RECORDS):
    base = ROLE_SALARY_BASE[job_roles[i]]
    edu_mult = 1 + (education_rank[i] - 1) * 0.10        # +10% per education tier
    tenure_mult = 1 + years_at_company[i] * 0.015         # +1.5% per year of tenure
    noise = rng.normal(1.0, 0.08)                          # +/-8% random noise
    salary = base * edu_mult * tenure_mult * noise
    monthly_salary.append(round(np.clip(salary, 25000, 150000), 2))
monthly_salary = np.array(monthly_salary)

# -----------------------------------------------
# 12. Distance from Home
# -----------------------------------------------
distance_from_home = rng.integers(1, 51, size=NUM_RECORDS)

# -----------------------------------------------
# 13. Attrition (Target Variable)
# -----------------------------------------------
# Logistic-style scoring for realistic correlations
attrition_logit = (
    -1.7                                                    # base (~16% rate)
    + 0.8  * (satisfaction_score < 2.5).astype(float)       # low satisfaction (increases)
    + 0.6  * (overtime == "Yes").astype(float)               # overtime (increases)
    - 0.5  * ((monthly_salary - 25000) / 125000)            # higher salary (decreases)
    + 0.4  * ((distance_from_home - 1) / 49)                # far from home (increases)
    - 0.4  * ((work_life_balance - 1) / 4)                  # good WLB (decreases)
    - 0.3  * (marital_statuses == "Married").astype(float)   # married (decreases)
    - 0.3  * np.clip(years_at_company / 30, 0, 1)           # longer tenure (decreases)
)

attrition_prob = 1 / (1 + np.exp(-attrition_logit))
attrition = np.array(["Yes" if rng.random() < p else "No" for p in attrition_prob])

# -----------------------------------------------
# 14. Assemble DataFrame
# -----------------------------------------------
df = pd.DataFrame({
    "employee_id":          employee_ids,
    "age":                  ages,
    "gender":               genders,
    "department":           departments,
    "job_role":             job_roles,
    "education_level":      education_levels,
    "monthly_salary":       monthly_salary,
    "years_at_company":     years_at_company,
    "years_since_promotion": years_since_promotion,
    "num_projects":         num_projects,
    "avg_weekly_hours":     avg_weekly_hours,
    "overtime":             overtime,
    "satisfaction_score":   satisfaction_score,
    "performance_rating":   performance_rating,
    "work_life_balance":    work_life_balance,
    "distance_from_home":   distance_from_home,
    "marital_status":       marital_statuses,
    "attrition":            attrition,
})

# -----------------------------------------------
# 15. Summary & Export
# -----------------------------------------------
print("=" * 60)
print("  SYNTHETIC EMPLOYEE ATTRITION DATASET - SUMMARY")
print("=" * 60)
print(f"\n  Total records : {len(df)}")
print(f"  Columns       : {df.shape[1]}")
print(f"\n  Attrition rate: {df['attrition'].value_counts(normalize=True)['Yes']:.1%}")

print("\n-- Numeric Column Statistics --")
print(df.describe().round(2).to_string())

print("\n-- Categorical Column Value Counts --")
for col in ["gender", "department", "job_role", "education_level",
            "overtime", "marital_status", "attrition"]:
    print(f"\n  {col}:")
    counts = df[col].value_counts()
    for val, cnt in counts.items():
        print(f"    {val:20s} : {cnt:4d}  ({cnt/len(df):.1%})")

print(f"\n  Saving to '{OUTPUT_FILE}' ...")
df.to_csv(OUTPUT_FILE, index=False)
print(f"  [OK] Saved {len(df)} records to {OUTPUT_FILE}")
print("=" * 60)
