"""
Hyperparameter tuning using CROSS-VALIDATION (prevents overfitting)
Test data is NOT used for tuning - only for final evaluation
"""

import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
from datetime import datetime

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("documentation", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TUNE_SAMPLE_SIZE = 500000

print("Loading data...")
df = pd.read_csv("datasets/cleaned_data.csv")
print(f"Full data: {len(df):,} rows")

df['Severity'] = df['Severity'] - 1

exclude_cols = ['Severity', 'City', 'County', 'Start_Lat', 'Start_Lng', 'Description']
feature_cols = [c for c in df.columns if c not in exclude_cols]

categorical_cols = ['State', 'Weather_Condition', 'TimeOfDay', 'Season', 'Sunrise_Sunset']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[feature_cols].values
y = df['Severity'].values

print(f"Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(X_train):,} rows")
print(f"Test: {len(X_test):,} rows (held out for final evaluation only)")

X_tune = X_train[:TUNE_SAMPLE_SIZE]
y_tune = y_train[:TUNE_SAMPLE_SIZE]
print(f"Tuning sample: {len(X_tune):,} rows (from training data)")

param_grid = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 150, 200],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist'
)

print("Starting cross-validation tuning...")
print("(Test data is NOT used - no overfitting risk)")

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_search.fit(X_tune, y_tune)

best_params = random_search.best_params_
print(f"\nBest parameters: {best_params}")

print("\nTraining final model on full training data...")
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist',
    **best_params
)

final_model.fit(X_train, y_train)

print("Evaluating on HELD-OUT test data (never seen before)...")
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy*100:.2f}%")

joblib.dump(final_model, "outputs/models/tuned_model.pkl")
print("Model saved: outputs/models/tuned_model.pkl")


# =========================================================
# SAVE REPORT
# =========================================================
report = f"""# Hyperparameter Tuning Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

|       Parameter       |        Value         |
|-----------------------|----------------------|
| Training rows         | {len(X_train):,}     |
| Testing rows          | {len(X_test):,}      |
| Tuning sample         | {TUNE_SAMPLE_SIZE:,} |
| Cross-validation folds| 5                    |
| Combinations tested   | 20                   |

## Best Parameters Found

|    Parameter    |                   Value                      |
|-----------------|----------------------------------------------|
| max_depth       | {best_params.get('max_depth', 'N/A')}        |
| learning_rate   | {best_params.get('learning_rate', 'N/A')}    |
| n_estimators    | {best_params.get('n_estimators', 'N/A')}     |
| subsample       | {best_params.get('subsample', 'N/A')}        |
| colsample_bytree| {best_params.get('colsample_bytree', 'N/A')} |
| min_child_weight| {best_params.get('min_child_weight', 'N/A')} |

## Results

|      Model    |      Accuracy       |
|---------------|---------------------|
| Tuned XGBoost | {accuracy*100:.2f}% |

## Output Files

- **Model:** `outputs/models/tuned_model.pkl`
- **Report:** `documentation/tuning_report.md`

## Next Step

Run `compare_default_tuned.py` to compare with default model.
"""

with open("documentation/tuning_report.md", "w") as f:
    f.write(report)

print("\nReport saved: documentation/tuning_report.md")