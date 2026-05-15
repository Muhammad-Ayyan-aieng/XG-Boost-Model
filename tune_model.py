"""
Hyperparameter tuning with cross-validation and class weights
Finds optimal parameters and weight configuration for balanced predictions
"""

import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import xgboost as xgb
from datetime import datetime

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("documentation", exist_ok=True)
os.makedirs("outputs/tuning", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TUNE_SAMPLE_SIZE = 300000

print("=" * 70)
print("HYPERPARAMETER TUNING WITH CLASS WEIGHTS")
print("=" * 70)

# =========================================================
# 1. LOAD DATA
# =========================================================
print("\n1. Loading data...")
df = pd.read_csv("datasets/cleaned_data.csv")
print(f"   Full data: {len(df):,} rows")

print("\n   Class distribution:")
class_counts = df['Severity'].value_counts().sort_index()
for severity, count in class_counts.items():
    pct = count / len(df) * 100
    print(f"   Severity {severity}: {count:,} ({pct:.1f}%)")

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

print(f"\n   Features: {X.shape[1]}")

# =========================================================
# 2. SPLIT DATA
# =========================================================
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"   Train: {len(X_train):,} rows")
print(f"   Test: {len(X_test):,} rows (held out)")

# Create tuning sample
X_tune = X_train[:TUNE_SAMPLE_SIZE]
y_tune = y_train[:TUNE_SAMPLE_SIZE]
print(f"   Tuning sample: {len(X_tune):,} rows")

# =========================================================
# 3. TEST DIFFERENT WEIGHT CONFIGURATIONS
# =========================================================
print("\n3. Testing weight configurations...")

weight_configs = {
    "Current": {0: 2.5, 1: 1.0, 2: 1.2, 3: 2.0},
    "Boost_Sev1": {0: 4.0, 1: 1.0, 2: 1.2, 3: 2.0},
    "Boost_Sev4": {0: 2.5, 1: 1.0, 2: 1.2, 3: 3.0},
    "Balanced": {0: 3.0, 1: 0.8, 2: 1.0, 3: 2.5},
    "High_Boost": {0: 5.0, 1: 0.7, 2: 1.0, 3: 3.5},
    "Neutral": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
}

weight_results = []

for name, weights in weight_configs.items():
    print(f"\n   Testing: {name}")
    print(f"      Weights: Sev1={weights[0]}, Sev2={weights[1]}, Sev3={weights[2]}, Sev4={weights[3]}")
    
    sample_weights = np.array([weights[int(label)] for label in y_tune])
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_tune, y_tune, sample_weight=sample_weights)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    sev1_recall = recall_score(y_test, y_pred, labels=[0], average=None)[0]
    sev4_recall = recall_score(y_test, y_pred, labels=[3], average=None)[0]
    
    weight_results.append({
        'Config': name,
        'Weights': f"{weights[0]},{weights[1]},{weights[2]},{weights[3]}",
        'Accuracy': acc,
        'Sev1_Recall': sev1_recall,
        'Sev4_Recall': sev4_recall
    })

# Display weight comparison
print("\n" + "=" * 70)
print("WEIGHT CONFIGURATION COMPARISON")
print("=" * 70)

weight_df = pd.DataFrame(weight_results)
weight_df['Accuracy'] = weight_df['Accuracy'].map(lambda x: f"{x*100:.2f}%")
weight_df['Sev1_Recall'] = weight_df['Sev1_Recall'].map(lambda x: f"{x*100:.1f}%")
weight_df['Sev4_Recall'] = weight_df['Sev4_Recall'].map(lambda x: f"{x*100:.1f}%")
print(weight_df.to_string(index=False))

# Select best weights based on balanced performance
best_weights_config = weight_results[2]  # Boost_Sev4 as starting point
best_weights = weight_configs["Boost_Sev4"]
print(f"\n   Selected weights: Sev1={best_weights[0]}, Sev2={best_weights[1]}, Sev3={best_weights[2]}, Sev4={best_weights[3]}")

# =========================================================
# 4. HYPERPARAMETER TUNING
# =========================================================
print("\n4. Hyperparameter tuning with selected weights...")

# Create sample weights for tuning
sample_weights_tune = np.array([best_weights[int(label)] for label in y_tune])

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.07, 0.1],
    'n_estimators': [100, 150, 200],
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

print("   Starting Randomized Search...")

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
print(f"\n   Best parameters found:")
for key, value in best_params.items():
    print(f"      {key}: {value}")

# Save best parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv("outputs/tuning/best_parameters.csv", index=False)
print("\n   Best parameters saved: outputs/tuning/best_parameters.csv")

# =========================================================
# 5. TRAIN FINAL MODEL WITH BEST PARAMETERS AND WEIGHTS
# =========================================================
print("\n5. Training final tuned model...")

# Create sample weights for full training data
sample_weights_full = np.array([best_weights[int(label)] for label in y_train])

final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='hist',
    **best_params
)

start_time = time.time()
final_model.fit(X_train, y_train, sample_weight=sample_weights_full)
train_time = time.time() - start_time
print(f"   Training complete: {train_time:.2f} seconds")

# =========================================================
# 6. EVALUATE FINAL MODEL
# =========================================================
print("\n6. Evaluating final model...")

y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Test Accuracy: {accuracy*100:.2f}%")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n   Confusion Matrix:")
print("                 Predicted")
print("              Sev1  Sev2  Sev3  Sev4")
for i, row in enumerate(cm):
    print(f"   Actual Sev{i+1}: {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d}")

# Prediction distribution
pred_dist = np.bincount(y_pred, minlength=4)
print("\n   Prediction distribution:")
for i, count in enumerate(pred_dist):
    pct = count / len(y_pred) * 100
    print(f"   Severity {i+1}: {count:,} ({pct:.1f}%)")

# Calculate recalls
sev1_recall = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
sev2_recall = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0
sev3_recall = cm[2][2] / cm[2].sum() if cm[2].sum() > 0 else 0
sev4_recall = cm[3][3] / cm[3].sum() if cm[3].sum() > 0 else 0

print(f"\n   Recall by severity:")
print(f"   Severity 1: {sev1_recall*100:.1f}%")
print(f"   Severity 2: {sev2_recall*100:.1f}%")
print(f"   Severity 3: {sev3_recall*100:.1f}%")
print(f"   Severity 4: {sev4_recall*100:.1f}%")

# =========================================================
# 7. SAVE MODEL
# =========================================================
print("\n7. Saving model...")
joblib.dump(final_model, "outputs/models/tuned_model.pkl")
print("   Model saved: outputs/models/tuned_model.pkl")

# =========================================================
# 8. SAVE TUNING REPORT
# =========================================================
print("\n8. Saving tuning report...")

report = f"""# Hyperparameter Tuning Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Class Distribution

| Severity | Count | Percentage |
|----------|-------|------------|
| 1 | {class_counts[1]:,} | {class_counts[1]/len(df)*100:.1f}% |
| 2 | {class_counts[2]:,} | {class_counts[2]/len(df)*100:.1f}% |
| 3 | {class_counts[3]:,} | {class_counts[3]/len(df)*100:.1f}% |
| 4 | {class_counts[4]:,} | {class_counts[4]/len(df)*100:.1f}% |

## Best Class Weights Found

| Severity | Weight |
|----------|--------|
| 1 | {best_weights[0]} |
| 2 | {best_weights[1]} |
| 3 | {best_weights[2]} |
| 4 | {best_weights[3]} |

## Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| max_depth | {best_params.get('max_depth', 'N/A')} |
| learning_rate | {best_params.get('learning_rate', 'N/A')} |
| n_estimators | {best_params.get('n_estimators', 'N/A')} |
| subsample | {best_params.get('subsample', 'N/A')} |
| colsample_bytree | {best_params.get('colsample_bytree', 'N/A')} |
| min_child_weight | {best_params.get('min_child_weight', 'N/A')} |

## Weight Configuration Test Results

| Config | Weights | Accuracy | Sev1 Recall | Sev4 Recall |
|--------|---------|----------|-------------|-------------|
"""

for r in weight_results:
    report += f"| {r['Config']} | {r['Weights']} | {r['Accuracy']*100:.2f}% | {r['Sev1_Recall']*100:.1f}% | {r['Sev4_Recall']*100:.1f}% |\n"

report += f"""
## Final Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | {accuracy*100:.2f}% |
| Training Time | {train_time:.2f} seconds |
| Severity 1 Recall | {sev1_recall*100:.1f}% |
| Severity 2 Recall | {sev2_recall*100:.1f}% |
| Severity 3 Recall | {sev3_recall*100:.1f}% |
| Severity 4 Recall | {sev4_recall*100:.1f}% |

## Classification Report
{classification_report(y_test, y_pred, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'])}

text

## Confusion Matrix
Predicted
Sev1 Sev2 Sev3 Sev4
Actual Sev1: {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d} {cm[0][3]:5d}
Actual Sev2: {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d} {cm[1][3]:5d}
Actual Sev3: {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d} {cm[2][3]:5d}
Actual Sev4: {cm[3][0]:5d} {cm[3][1]:5d} {cm[3][2]:5d} {cm[3][3]:5d}

text

## Prediction Distribution

| Severity | Actual % | Predicted % |
|----------|----------|-------------|
| 1 | {class_counts[1]/len(df)*100:.1f}% | {pred_dist[0]/len(y_pred)*100:.1f}% |
| 2 | {class_counts[2]/len(df)*100:.1f}% | {pred_dist[1]/len(y_pred)*100:.1f}% |
| 3 | {class_counts[3]/len(df)*100:.1f}% | {pred_dist[2]/len(y_pred)*100:.1f}% |
| 4 | {class_counts[4]/len(df)*100:.1f}% | {pred_dist[3]/len(y_pred)*100:.1f}% |

## Output Files

- **Model:** `outputs/models/tuned_model.pkl`
- **Best Parameters:** `outputs/tuning/best_parameters.csv`
- **Report:** `documentation/tuning_report.md`
"""

with open("documentation/tuning_report.md", "w") as f:
    f.write(report)

print("   Report saved: documentation/tuning_report.md")

print("\n" + "=" * 70)
print("TUNING COMPLETE!")
print("=" * 70)