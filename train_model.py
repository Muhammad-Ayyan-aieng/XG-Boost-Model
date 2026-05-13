"""
Train best model (XGBoost) on full 6.7M dataset
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
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("documentation", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("Loading best model info...")
compare_df = pd.read_csv("outputs/results/model_comparison.csv")
best_model_name = compare_df.iloc[0]['model']
print(f"Best model: {best_model_name}")

print("Loading full data...")
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
print(f"Classes: {np.unique(y)}")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(X_train):,} rows")
print(f"Test: {len(X_test):,} rows")

print("Training XGBoost on full data...")
start = time.time()

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)
train_time = time.time() - start
print(f"Training complete: {train_time:.2f} seconds")

print("Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "outputs/models/final_model.pkl")
print("Model saved: outputs/models/final_model.pkl")

report_path = "documentation/training_report.md"
with open(report_path, "w") as f:
    f.write("# Model Training Report\n\n")
    f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## Training Configuration\n\n")
    f.write(f"|    Parameter     |      Value       |\n")
    f.write(f"|------------------|------------------|\n")
    f.write(f"| Model            | XGBoost          |\n")
    f.write(f"| Training rows    | {len(X_train):,} |\n")
    f.write(f"| Testing rows     | {len(X_test):,}  |\n")
    f.write(f"| Features         | {X.shape[1]}  |\n")
    f.write(f"| Train/Test split | {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}% |\n\n")
    f.write("## Performance\n\n")
    f.write(f"|      Metric    |          Value           |\n")
    f.write(f"|----------------|--------------------------|\n")
    f.write(f"| Test Accuracy  | {accuracy*100:.2f}%      |\n")
    f.write(f"| Training Time  | {train_time:.2f} seconds |\n\n")
    f.write("## Classification Report\n\n```\n")
    f.write(classification_report(y_test, y_pred, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4']))
    f.write("\n```\n\n")
    f.write("## Output\n\n")
    f.write("- **Model file:** `outputs/models/final_model.pkl`\n")
    f.write("- **Ready for API deployment**\n")

print(f"Report saved: {report_path}")