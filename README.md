# Customer Churn Prediction

An end-to-end machine learning pipeline that predicts whether a telecom customer will churn (leave), using the IBM Telco Customer Churn dataset (7,043 customers, 21 features).

## Live Demo
Deployed with Streamlit: `[add your share.streamlit.io link here after deploying]`

## Problem
Customer churn directly impacts revenue. This project identifies at-risk customers in advance so a business can take proactive retention action, and explains *why* each prediction was made.

## Workflow
1. **Data Cleaning** — fixed hidden blank values in `TotalCharges`, encoded target variable
2. **EDA** — found churn strongly correlates with contract type (42.7% churn on month-to-month vs 2.8% on two-year), tenure, and monthly charges
3. **Feature Engineering** — Label Encoding (binary features) + One-Hot Encoding (multi-category features) + StandardScaler (numeric features)
4. **Class Imbalance Handling** — SMOTE (Synthetic Minority Oversampling), applied correctly inside a `Pipeline` to avoid data leakage during cross-validation
5. **Model Comparison** — Logistic Regression, Random Forest, XGBoost
6. **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation, optimized for Recall
7. **Deployment** — Streamlit web app for live predictions

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 73.2% | 49.6% | 70.1% | 58.1% | 82.2% |
| Random Forest (baseline) | 76.0% | 54.0% | 62.6% | 58.0% | 81.9% |
| XGBoost | 75.6% | 53.3% | 65.5% | 58.8% | 81.1% |
| **Random Forest (tuned, final)** | **74.2%** | — | **79.9%** | **62.2%** | — |

**Final model chosen:** Tuned Random Forest — optimized for Recall since missing a real churner (false negative) is costlier for the business than a false alarm.

## Top Churn Drivers (Feature Importance)
1. Tenure (customer's time with company)
2. Total Charges
3. Monthly Charges
4. Payment Method (Electronic check)
5. Contract Type (Two year contract reduces churn risk sharply)

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Streamlit

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit joblib

# Train the model (recreates churn_deployment_bundle.pkl)
python train_model.py

# Launch the interactive prediction app
streamlit run app.py
```

## Project Structure
```
├── telco_churn.csv               # Dataset
├── train_model.py                # Full pipeline: EDA -> preprocessing -> training -> tuning
├── app.py                        # Streamlit web app for live predictions
├── churn_deployment_bundle.pkl   # Saved model + preprocessing objects
└── README.md
```

## Key Learnings
- Diagnosed and fixed a data leakage bug: applying SMOTE before cross-validation inflated validation scores by letting synthetic rows "leak" information from validation folds. Fixed by moving SMOTE inside an `imblearn.Pipeline`, which reduced the train-test overfitting gap from 0.236 to 0.021.
- Compared models on multiple metrics, not just accuracy — accuracy alone is misleading on this dataset due to class imbalance (73.5% vs 26.5%).