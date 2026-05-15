"""
Train XGBoost on full 6.7M dataset with BALANCED CLASS WEIGHTS
Output: outputs/models/final_model.pkl
        documentation/training_report.md
"""

import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("documentation", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("=" * 60)
print("TRAINING XGBOOST WITH BALANCED CLASS WEIGHTS")
print("=" * 60)

print("\n1. Loading data...")
df = pd.read_csv("datasets/cleaned_data.csv")
print(f"   Full data: {len(df):,} rows")

# Show class distribution BEFORE remapping
print("\n   Original class distribution (1,2,3,4):")
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

print(f"\n2. Features: {X.shape[1]}")

print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"   Train: {len(X_train):,} rows")
print(f"   Test: {len(X_test):,} rows")

# =========================================================
# BALANCED CLASS WEIGHTS (MANUAL - FIXED FOR LOW BIAS)
# =========================================================
print("\n4. Setting balanced class weights...")

# MANUAL BALANCED WEIGHTS - Test these values
# These weights are designed to prevent over-prediction of high severity
class_weights_dict = {
    0: 2.5,   # Severity 1 (Minor) - moderate emphasis
    1: 1.0,   # Severity 2 (Moderate) - baseline
    2: 1.2,   # Severity 3 (Serious) - slight emphasis
    3: 2.0    # Severity 4 (Severe) - reduced from 9.46 to 2.0
}

sample_weights = np.array([class_weights_dict[int(label)] for label in y_train])

print("   Class weights used:")
for i in range(4):
    print(f"   Severity {i+1}: {class_weights_dict[i]:.2f}")

# =========================================================
# TRAIN WITH OPTIMIZED PARAMETERS
# =========================================================
print("\n5. Training XGBoost with balanced weights...")
start = time.time()

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=200,      # Balanced number of trees
    max_depth=6,           # Shallower trees to prevent overfitting
    learning_rate=0.07,    # Medium learning rate
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train, sample_weight=sample_weights)
train_time = time.time() - start
print(f"   Training complete: {train_time:.2f} seconds")

# =========================================================
# EVALUATE
# =========================================================
print("\n6. Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Test accuracy: {accuracy*100:.2f}%")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n   Confusion Matrix:")
print("                 Predicted")
print("              Sev1  Sev2  Sev3  Sev4")
for i, row in enumerate(cm):
    print(f"   Actual Sev{i+1}: {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d}")

# Check prediction distribution
pred_dist = np.bincount(y_pred, minlength=4)
print("\n   Prediction distribution:")
for i, count in enumerate(pred_dist):
    pct = count / len(y_pred) * 100
    print(f"   Severity {i+1}: {count:,} ({pct:.1f}%)")

# Check average predictions
print(f"\n   Average predicted severity: {y_pred.mean() + 1:.2f}")
print(f"   Average actual severity: {y_test.mean() + 1:.2f}")

# =========================================================
# SAVE MODEL
# =========================================================
print("\n7. Saving model...")
joblib.dump(model, "outputs/models/final_model.pkl")
print("   Model saved: outputs/models/final_model.pkl")

# =========================================================
# SAVE REPORT
# =========================================================
print("\n8. Saving report...")
report_path = "documentation/training_report.md"
with open(report_path, "w") as f:
    f.write("# Model Training Report\n\n")
    f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## Class Distribution\n\n")
    f.write("| Severity | Count | Percentage |\n")
    f.write("|----------|-------|------------|\n")
    for severity, count in class_counts.items():
        pct = count / len(df) * 100
        f.write(f"| {severity} | {count:,} | {pct:.1f}% |\n")
    
    f.write("\n## Class Weights Used\n\n")
    f.write("| Severity | Weight |\n")
    f.write("|----------|--------|\n")
    for i in range(4):
        f.write(f"| {i+1} | {class_weights_dict[i]:.2f} |\n")
    
    f.write("\n## Training Configuration\n\n")
    f.write("| Parameter | Value |\n")
    f.write("|-----------|-------|\n")
    f.write(f"| Model | XGBoost |\n")
    f.write(f"| Training rows | {len(X_train):,} |\n")
    f.write(f"| Testing rows | {len(X_test):,} |\n")
    f.write(f"| Features | {X.shape[1]} |\n")
    f.write(f"| n_estimators | 200 |\n")
    f.write(f"| max_depth | 6 |\n")
    f.write(f"| learning_rate | 0.07 |\n")
    
    f.write("\n## Performance\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Test Accuracy | {accuracy*100:.2f}% |\n")
    f.write(f"| Training Time | {train_time:.2f} seconds |\n\n")
    
    f.write("## Prediction Distribution\n\n")
    f.write("| Severity | Actual % | Predicted % |\n")
    f.write("|----------|----------|-------------|\n")
    for i in range(4):
        actual_pct = (y_test == i).sum() / len(y_test) * 100
        pred_pct = pred_dist[i] / len(y_pred) * 100
        f.write(f"| {i+1} | {actual_pct:.1f}% | {pred_pct:.1f}% |\n")
    
    f.write("\n## Classification Report\n\n```\n")
    f.write(classification_report(y_test, y_pred, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4']))
    f.write("\n```\n\n")
    
    f.write("## Confusion Matrix\n\n```\n")
    f.write("                 Predicted\n")
    f.write("              Sev1  Sev2  Sev3  Sev4\n")
    for i, row in enumerate(cm):
        f.write(f"Actual Sev{i+1}: {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d}\n")
    f.write("```\n\n")
    
    f.write("## Output Files\n\n")
    f.write("- `outputs/models/final_model.pkl`\n")
 
print(f"   Report saved: {report_path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)