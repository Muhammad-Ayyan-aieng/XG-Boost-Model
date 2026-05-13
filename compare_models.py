"""
Compare 6 models on sample data (1M rows)
Outputs: 
  - outputs/results/model_comparison.csv
  - documentation/model_comparison_report.md
  - outputs/models/*.pkl (all 6 models)
"""

import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)
os.makedirs("documentation", exist_ok=True)

SAMPLE_SIZE = 1000000
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("Loading data...")
df = pd.read_csv("datasets/cleaned_data.csv")
print(f"Full data: {len(df):,} rows")

df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
print(f"Sample: {len(df):,} rows")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(X_train):,} rows")
print(f"Test: {len(X_test):,} rows")

models = {
    'logistic_regression': LogisticRegression(max_iter=500, random_state=RANDOM_STATE, n_jobs=-1),
    'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'random_forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    'xgboost': xgb.XGBClassifier(objective='multi:softmax', num_class=4, n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
    'lightgbm': lgb.LGBMClassifier(objective='multiclass', num_class=4, n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
    'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=RANDOM_STATE, early_stopping=True)
}

results = []

print("\nTraining models...")

for name, model in models.items():
    print(f"  {name}...", end=" ", flush=True)
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    
    model_path = f"outputs/models/{name}.pkl"
    joblib.dump(model, model_path)
    
    results.append({
        'model': name,
        'accuracy': acc,
        'train_time_sec': round(train_time, 2),
        'inference_time_ms': round(infer_time * 1000, 2),
        'model_file': model_path
    })
    
    print(f"acc={acc:.4f} ({acc*100:.2f}%)")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('accuracy', ascending=False)

results_df.to_csv("outputs/results/model_comparison.csv", index=False)

best_model = results_df.iloc[0]['model']
best_accuracy = results_df.iloc[0]['accuracy']

print(f"\nBest model: {best_model} ({best_accuracy*100:.2f}%)")

with open("documentation/model_comparison_report.md", "w") as f:
    f.write("# Model Comparison Report\n\n")
    f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Sample Size:** {SAMPLE_SIZE:,} rows\n\n")
    f.write(f"**Train/Test Split:** {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%\n\n")
    
    f.write("## Results\n\n")
    f.write("| Model | Accuracy | Train Time (s) | Inference Time (ms) |\n")
    f.write("|-------|----------|----------------|---------------------|\n")
    
    for _, row in results_df.iterrows():
        f.write(f"| {row['model']} | {row['accuracy']*100:.2f}% | {row['train_time_sec']} | {row['inference_time_ms']} |\n")
    
    f.write("\n## Best Model\n\n")
    f.write(f"**{best_model}** achieved the highest accuracy with **{best_accuracy*100:.2f}%**.\n\n")

print(f"\nResults saved to:")
print(f"  - outputs/results/model_comparison.csv")
print(f"  - documentation/model_comparison_report.md")
print(f"  - outputs/models/*.pkl ({len(models)} models)")