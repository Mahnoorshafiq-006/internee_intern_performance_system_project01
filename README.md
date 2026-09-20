# internee_intern_performance_system_project01
# Intern Performance Prediction System

A machine learning regression project that predicts intern performance scores and classifies interns into performance tiers — **Excel**, **Average**, or **Struggle** — based on task completion time, feedback ratings, and attendance.

## Overview

This project analyzes a dataset of 10,000 intern records to predict `Performance_Score` and identify which interns are likely to excel or need additional support. It combines data cleaning, feature engineering, model comparison, and visualization into a complete end-to-end pipeline.

## Dataset

| Column | Description |
|---|---|
| `Intern_ID` | Unique identifier for each intern |
| `Completion_Time` | Time taken to complete tasks |
| `Feedback_Rating` | Supervisor feedback rating |
| `Attendance` | Attendance percentage |
| `Performance_Score` | Target variable — overall performance score |

## Project Pipeline

1. **Data Loading** — Load the dataset with pandas
2. **Data Cleaning**
   - Median imputation for missing values in `Completion_Time`, `Feedback_Rating`, and `Attendance`
   - Attendance values corrected and clipped to a valid 0–100 range
3. **Feature Engineering**
   - Derived `Efficiency_Index` = `Feedback_Rating / Completion_Time`, capturing quality of output relative to time spent
4. **Train/Test Split** — 80% training, 20% testing
5. **Modeling** — Random Forest Regressor and XGBoost Regressor, trained and tuned
6. **Evaluation** — Compared using MAE, RMSE, and R² score
7. **Classification** — Interns ranked by predicted score and split into tiers using percentile thresholds:
   - Top 30% → **Excel**
   - Bottom 30% → **Struggle**
   - Middle 40% → **Average**
8. **Visualization** — Bar chart of interns by performance category
9. **Report Export** — Final results saved as a CSV report

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 7.56 | 9.39 | 0.61 |
| **XGBoost** | **7.40** | **9.26** | **0.62** |

Feature importance analysis showed `Completion_Time` and `Feedback_Rating` as the strongest predictors of performance, with `Attendance` playing a smaller role.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib, Seaborn

## Project Structure

```
├── intern_dataset.csv                     # Raw dataset
├── intern_performance_model.py            # Main pipeline script
├── intern_performance_report_final.csv    # Output: per-intern predictions & category
├── performance_category_distribution.png  # Output: category distribution chart
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python intern_performance_model.py
```

## Output

- Console output: model evaluation metrics and feature importance
- `performance_category_distribution.png` — visual breakdown of intern performance tiers
- `intern_performance_report_final.csv` — final report with predicted scores and categories per intern


