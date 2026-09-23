"""
CUSTOMER CHURN PREDICTION - End-to-End ML Pipeline
=====================================================
Run this with: python train_model.py
(Just needs: pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib)

This script covers the full ML workflow:
1. Data Loading  2. EDA  3. Preprocessing  4. Train-Test Split
5. SMOTE (class imbalance)  6. Model Training (3 models)
7. Evaluation  8. Feature Importance  9. Hyperparameter Tuning
10. Save final deployment-ready model
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

print("="*60)
print("STEP 1-2: LOAD DATA")
print("="*60)
df = pd.read_csv('telco_churn.csv')
print(f"Dataset shape: {df.shape}")

print("\n" + "="*60)
print("STEP 3: EDA")
print("="*60)
print("\nChurn distribution:")
print(df['Churn'].value_counts(normalize=True) * 100)
print("\nChurn rate by Contract type:")
print(pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100)

print("\n" + "="*60)
print("STEP 4: PREPROCESSING")
print("="*60)
# Fix TotalCharges (hidden blanks -> numeric)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Label Encoding for binary columns
binary_cols = [c for c in df.select_dtypes(include='object').columns if df[c].nunique() == 2]
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-Hot Encoding for multi-category columns
multi_cols = list(df.select_dtypes(include='object').columns)
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Feature Scaling
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(f"Final feature count: {df.shape[1] - 1}")

print("\n" + "="*60)
print("STEP 5: TRAIN-TEST SPLIT")
print("="*60)
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

print("\n" + "="*60)
print("STEP 6-7: TRAIN & EVALUATE BASELINE MODELS")
print("="*60)
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = []
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "="*60)
print("STEP 8: FEATURE IMPORTANCE")
print("="*60)
rf_model = models['Random Forest']
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance.head(10).to_string(index=False))

print("\n" + "="*60)
print("STEP 9: HYPERPARAMETER TUNING (leak-free SMOTE inside pipeline)")
print("="*60)
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [6, 10],
    'rf__min_samples_leaf': [5, 10]
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid.fit(X_train, y_train)  # NOTE: uses ORIGINAL non-SMOTE'd data; SMOTE runs inside CV folds

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Best params: {grid.best_params_}")
print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Final Test Recall  : {recall_score(y_test, y_pred):.3f}")
print(f"Final Test F1       : {f1_score(y_test, y_pred):.3f}")

print("\n" + "="*60)
print("STEP 10: SAVE DEPLOYMENT BUNDLE")
print("="*60)
bundle = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_columns': X_train.columns.tolist(),
    'num_cols': num_cols,
    'multi_cols': multi_cols,
    'binary_cols': binary_cols
}
joblib.dump(bundle, 'churn_deployment_bundle.pkl')
print("Saved: churn_deployment_bundle.pkl")
print("\nDONE. Now run: streamlit run app.py")