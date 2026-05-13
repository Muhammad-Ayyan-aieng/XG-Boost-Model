"""
Hyperparameter tuning using CROSS-VALIDATION (prevents overfitting)
Test data is NOT used for tuning - only for final evaluation
"""

import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from datetime import datetime

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("documentation", exist_ok=True)
os.makedirs("outputs/tuning", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TUNE_SAMPLE_SIZE = 500000

print("=" * 60)
print("HYPERPARAMETER TUNING WITH CLASS WEIGHTS")
print("=" * 60)

print("\n1. Loading data...")
df = pd.read_csv("datasets/cleaned_data.csv")
print(f"   Full data: {len(df):,} rows")

print("\n   Class distribution:")
class_counts = df['Severity'].value_counts().sort_index()
for sev, count in class_counts.items():
    print(f"   Severity {sev}: {count:,} ({count/len(df)*100:.1f}%)")

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

print(f"\n2. Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"   Train: {len(X_train):,} rows")
print(f"   Test: {len(X_test):,} rows (held out)")

# Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
sample_weights = np.array([class_weights[int(label)] for label in y_train])

print("\n3. Class weights:")
for i, w in enumerate(class_weights):
    print(f"   Severity {i+1}: {w:.4f}")

X_tune = X_train[:TUNE_SAMPLE_SIZE]
y_tune = y_train[:TUNE_SAMPLE_SIZE]
sample_weights_tune = sample_weights[:TUNE_SAMPLE_SIZE]
print(f"\n4. Tuning sample: {len(X_tune):,} rows")

param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [200, 300, 400],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist'
)

print("\n5. Starting cross-validation tuning...")
print("   (Test data is NOT used - no overfitting risk)")

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    n_iter=15,
    cv=3,
    scoring='accuracy',
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_search.fit(X_tune, y_tune, sample_weight=sample_weights_tune)

best_params = random_search.best_params_
print(f"\n6. Best parameters found:")
for key, value in best_params.items():
    print(f"   {key}: {value}")

# Save best parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv("outputs/tuning/best_parameters.csv", index=False)
print("\n   Best parameters saved: outputs/tuning/best_parameters.csv")

print("\n7. Training final model on full training data...")
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist',
    **best_params
)

final_model.fit(X_train, y_train, sample_weight=sample_weights)

print("\n8. Evaluating on HELD-OUT test data...")
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Test accuracy: {accuracy*100:.2f}%")

print("\n9. Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4']))

cm = confusion_matrix(y_test, y_pred)
print("\n10. Confusion Matrix:")
print(cm)

# Save model
joblib.dump(final_model, "outputs/models/tuned_model.pkl")
print("\n11. Model saved: outputs/models/tuned_model.pkl")

# =========================================================
# SAVE REPORT
# =========================================================
print("\n12. Saving report...")

report = f"""# Hyperparameter Tuning Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Class Distribution

| Severity | Count | Percentage |
|----------|-------|------------|
| 1 | {class_counts[1]:,} | {class_counts[1]/len(df)*100:.1f}% |
| 2 | {class_counts[2]:,} | {class_counts[2]/len(df)*100:.1f}% |
| 3 | {class_counts[3]:,} | {class_counts[3]/len(df)*100:.1f}% |
| 4 | {class_counts[4]:,} | {class_counts[4]/len(df)*100:.1f}% |

## Class Weights Used

| Severity | Weight |
|----------|--------|
| 1 | {class_weights[0]:.4f} |
| 2 | {class_weights[1]:.4f} |
| 3 | {class_weights[2]:.4f} |
| 4 | {class_weights[3]:.4f} |

## Configuration

| Parameter | Value |
|-----------|-------|
| Training rows | {len(X_train):,} |
| Testing rows | {len(X_test):,} |
| Tuning sample | {TUNE_SAMPLE_SIZE:,} |
| Cross-validation folds | 3 |
| Combinations tested | 15 |

## Best Parameters Found

| Parameter | Value |
|-----------|-------|
| max_depth | {best_params.get('max_depth', 'N/A')} |
| learning_rate | {best_params.get('learning_rate', 'N/A')} |
| n_estimators | {best_params.get('n_estimators', 'N/A')} |
| subsample | {best_params.get('subsample', 'N/A')} |
| colsample_bytree | {best_params.get('colsample_bytree', 'N/A')} |
| min_child_weight | {best_params.get('min_child_weight', 'N/A')} |

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | {accuracy*100:.2f}% |

## Classification Report
{classification_report(y_test, y_pred, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'])}

text

## Confusion Matrix
{cm}

text

## Output Files

- **Model:** `outputs/models/tuned_model.pkl`
- **Best Parameters:** `outputs/tuning/best_parameters.csv`
- **Report:** `documentation/tuning_report.md`
"""

with open("documentation/tuning_report.md", "w") as f:
    f.write(report)

print("   Report saved: documentation/tuning_report.md")

print("\n" + "=" * 60)
print("TUNING COMPLETE!")
print("=" * 60)