"""
Train XGBoost on full 6.7M dataset with CLASS WEIGHTS for ALL severity levels
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
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("documentation", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("=" * 60)
print("TRAINING XGBOOST WITH CLASS WEIGHTS")
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
# CALCULATE CLASS WEIGHTS FOR ALL SEVERITIES
# =========================================================
print("\n4. Calculating class weights...")
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
sample_weights = np.array([class_weights[int(label)] for label in y_train])

print("   Class weights (higher = more important):")
for i, weight in enumerate(class_weights):
    print(f"   Severity {i+1}: {weight:.4f}")

# =========================================================
# TRAIN WITH OPTIMIZED PARAMETERS
# =========================================================
print("\n5. Training XGBoost with class weights...")
start = time.time()

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=300,      # More trees for better learning
    max_depth=8,           # Deeper trees for complex patterns
    learning_rate=0.05,    # Slower learning for better accuracy
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

# Show probability distribution for severe cases
print("\n7. Testing severe case prediction...")
test_severe = X_test[y_test == 3]  # Severity 4 cases
if len(test_severe) > 0:
    probas = model.predict_proba(test_severe[:10])
    print(f"   Sample of 10 actual Severity 4 cases:")
    for i, prob in enumerate(probas[:5]):
        print(f"   Case {i+1}: Sev1={prob[0]:.2f}, Sev2={prob[1]:.2f}, Sev3={prob[2]:.2f}, Sev4={prob[3]:.2f}")

# =========================================================
# SAVE MODEL
# =========================================================
print("\n8. Saving model...")
joblib.dump(model, "outputs/models/final_model.pkl")
print("   Model saved: outputs/models/final_model.pkl")

# =========================================================
# SAVE REPORT
# =========================================================
print("\n9. Saving report...")
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
    for i, weight in enumerate(class_weights):
        f.write(f"| {i+1} | {weight:.4f} |\n")
    
    f.write("\n## Training Configuration\n\n")
    f.write("| Parameter | Value |\n")
    f.write("|-----------|-------|\n")
    f.write(f"| Model | XGBoost |\n")
    f.write(f"| Training rows | {len(X_train):,} |\n")
    f.write(f"| Testing rows | {len(X_test):,} |\n")
    f.write(f"| Features | {X.shape[1]} |\n")
    f.write(f"| n_estimators | 300 |\n")
    f.write(f"| max_depth | 8 |\n")
    f.write(f"| learning_rate | 0.05 |\n")
    
    f.write("\n## Performance\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Test Accuracy | {accuracy*100:.2f}% |\n")
    f.write(f"| Training Time | {train_time:.2f} seconds |\n\n")
    
    f.write("## Classification Report\n\n```\n")
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
    f.write("- `outputs/xgboost_model.pkl`\n")

print(f"   Report saved: {report_path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)