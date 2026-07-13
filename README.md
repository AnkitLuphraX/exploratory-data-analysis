# Exploratory Data Analysis (EDA) Project

An in-depth exploratory analysis of an Employee Attrition dataset to uncover patterns, identify key factors driving employee turnover, and present actionable insights.

## 📌 About

This project analyzes a synthetic HR dataset containing **2000 employee records** with 18 features. The goal is to perform comprehensive EDA using statistical methods and visualizations to understand what drives employee attrition.

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** – data manipulation and analysis
- **NumPy** – numerical computations
- **Matplotlib** – static visualizations
- **Seaborn** – statistical charts and heatmaps
- **SciPy** – statistical hypothesis testing

## 📁 Project Structure

```
├── generate_dataset.py              # generates the synthetic HR dataset
├── eda_analysis.py                  # full EDA pipeline with visualizations
├── eda_report.py                    # structured insights report + dashboard
├── employee_attrition_raw.csv       # raw dataset (generated)
├── visualizations/                  # all saved charts and dashboard
│   ├── 01_attrition_distribution.png
│   ├── 02_department_analysis.png
│   ├── 03_salary_analysis.png
│   ├── 04_satisfaction_analysis.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_overtime_analysis.png
│   ├── 07_tenure_analysis.png
│   ├── 08_promotion_analysis.png
│   ├── 09_worklife_balance.png
│   ├── 10_pairplot.png
│   ├── 11_chi_square_results.png
│   ├── 12_top_factors.png
│   └── eda_summary_dashboard.png
├── requirements.txt
└── README.md
```

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/AnkitLuphraX/exploratory-data-analysis.git
   cd exploratory-data-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the raw dataset**
   ```bash
   python generate_dataset.py
   ```

4. **Run full EDA analysis**
   ```bash
   python eda_analysis.py
   ```

5. **Generate the insights report**
   ```bash
   python eda_report.py
   ```

## 📊 Key Features

### Statistical Analysis
- Descriptive statistics (mean, median, skewness, kurtosis)
- Point-biserial correlation for numerical features vs attrition
- Chi-square tests for categorical independence
- Ranked influencing factors analysis

### Visualizations Created
| # | Chart | Description |
|---|-------|-------------|
| 1 | Attrition Distribution | Pie/count plot of attrition breakdown |
| 2 | Department Analysis | Attrition rate comparison across departments |
| 3 | Salary Analysis | Box/violin plots – salary by attrition status |
| 4 | Satisfaction Analysis | Distribution of satisfaction scores by group |
| 5 | Correlation Heatmap | Relationships between all numerical features |
| 6 | Overtime Analysis | Overtime vs non-overtime attrition rates |
| 7 | Tenure Analysis | Years at company distribution by attrition |
| 8 | Promotion Analysis | Impact of years since last promotion |
| 9 | Work-Life Balance | Work-life balance scores across groups |
| 10 | Pairplot | Multi-feature scatter matrix colored by attrition |
| 11 | Chi-Square Results | Categorical feature independence test results |
| 12 | Top Factors | Ranked bar chart of influencing factors |

### Insights Report
A combined 6-panel dashboard image + console report with key findings and recommendations.

## 💡 Key Insights

- Overtime employees show significantly higher attrition rates compared to non-overtime peers
- Lower satisfaction scores correlate strongly with increased employee turnover
- Monthly salary plays a meaningful role in retention – leavers earn noticeably less
- Shorter-tenured employees are more likely to leave, highlighting early-career retention gaps
- Work-life balance impacts attrition decisions, with lower scores among departing employees
- Certain departments exhibit disproportionately higher attrition, suggesting localized issues

## 📝 What I Learned

- Applying statistical tests (chi-square, point-biserial correlation) to validate hypotheses
- Creating comprehensive visualizations to tell a data story
- Identifying multivariate relationships through pairplots and heatmaps
- Translating analytical findings into actionable business recommendations
- Building end-to-end EDA pipelines from data generation to reporting

## 📄 License

This project is for educational purposes as part of my internship coursework.
