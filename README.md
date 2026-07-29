# Customer Churn Prediction

Predicts whether a telecom customer is likely to churn, using historical customer data, and identifies the key factors that drive churn.

## Overview
This project builds and compares two classification models to predict customer churn, then interprets which features most influence the predictions — providing actionable insight for retention strategy.

## Dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers, 21 features (demographics, services subscribed, billing/contract info)
- Target: `Churn` (Yes/No) — customer churn rate ~26.5%

## Approach
1. Data cleaning (handled missing/malformed `TotalCharges` values)
2. Exploratory Data Analysis — churn patterns across contract type, tenure, monthly charges, internet service
3. Feature engineering — one-hot encoding categorical variables, train/test split (80/20, stratified)
4. Model training — Logistic Regression (baseline) and Random Forest (comparison)
5. Evaluation — precision, recall, F1-score, ROC-AUC
6. Feature importance analysis to interpret churn drivers

## Results

|       Model         | ROC-AUC | Precision (Churn) | Recall (Churn) | F1 (Churn) |

| Logistic Regression | 0.836   | 0.65              | 0.58           | 0.61       |
| Random Forest       | 0.819   | 0.63              | 0.50           | 0.56       |

**Logistic Regression outperformed Random Forest** on this dataset — both in ROC-AUC and in recall (catching actual churners), while also being simpler and more interpretable. This suggests churn in this dataset is driven by fairly linear relationships (e.g., tenure, charges, contract type) that Logistic Regression captures well.

## Key Churn Drivers (Random Forest Feature Importance)
1. Total Charges
2. Tenure
3. Monthly Charges
4. Contract type (Two-year contracts reduce churn risk)
5. Internet Service (Fiber optic customers churn more)
6. Payment Method (Electronic check associated with higher churn)

## Business Recommendations
- Logistic Regression is recommended as the production model — it's simpler, more interpretable, and catches more actual churners (58% recall) than Random Forest
- Focus retention efforts on new/short-tenure customers, especially in their first few months
- Encourage month-to-month customers to switch to longer-term contracts (discounts/incentives)
- Investigate why fiber optic and electronic check customers churn more — could be price, service quality, or friction in the payment experience
- Note: with 58% recall, the model still misses ~42% of churners — this could be improved with more features (e.g., customer support interactions, usage trends) in a future iteration

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Notebook
Full analysis, code, and visualizations: [`customer-churn-prediction.ipynb`](./customer-churn-prediction.ipynb.ipynb)
