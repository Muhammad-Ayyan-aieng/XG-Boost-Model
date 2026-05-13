"""
Visualizations for model results
Generates confusion matrix, feature importance, accuracy comparison
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

os.makedirs("outputs/figures", exist_ok=True)

print("Loading data...")
df = pd.read_csv("datasets/cleaned_data.csv")
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Loading models...")
default_model = joblib.load("outputs/models/final_model.pkl")
tuned_model = joblib.load("outputs/models/tuned_model.pkl")

y_pred_default = default_model.predict(X_test)
y_pred_tuned = tuned_model.predict(X_test)
default_acc = accuracy_score(y_test, y_pred_default)
tuned_acc = accuracy_score(y_test, y_pred_tuned)

# =========================================================
# FIGURE 1: CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sev 1', 'Sev 2', 'Sev 3', 'Sev 4'],
            yticklabels=['Sev 1', 'Sev 2', 'Sev 3', 'Sev 4'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Tuned XGBoost')
plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrix.png', dpi=150)
plt.close()
print("Saved: outputs/figures/confusion_matrix.png")

# =========================================================
# FIGURE 2: FEATURE IMPORTANCE
# =========================================================
importance = tuned_model.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(feat_imp['feature'], feat_imp['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/figures/feature_importance.png', dpi=150)
plt.close()
print("Saved: outputs/figures/feature_importance.png")

# =========================================================
# FIGURE 3: ACCURACY COMPARISON (Default vs Tuned)
# =========================================================
plt.figure(figsize=(6, 6))
bars = plt.bar(['Default', 'Tuned'], [default_acc*100, tuned_acc*100],
               color=['#3498db', '#2ecc71'], edgecolor='black')
plt.ylabel('Accuracy (%)')
plt.title('Default vs Tuned XGBoost')
plt.ylim(80, 86)

for bar, acc in zip(bars, [default_acc*100, tuned_acc*100]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{acc:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('outputs/figures/accuracy_comparison.png', dpi=150)
plt.close()
print("Saved: outputs/figures/accuracy_comparison.png")

# =========================================================
# FIGURE 4: SEVERITY DISTRIBUTION
# =========================================================
severity_counts = df['Severity'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
bars = plt.bar([1, 2, 3, 4], severity_counts.values, color='#3498db', edgecolor='black')
plt.xlabel('Severity Level')
plt.ylabel('Number of Accidents')
plt.title('Severity Distribution in Dataset')
plt.xticks([1, 2, 3, 4])

for bar, count in zip(bars, severity_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'{count:,}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('outputs/figures/severity_distribution.png', dpi=150)
plt.close()
print("Saved: outputs/figures/severity_distribution.png")

print("\nAll visualizations saved to outputs/figures/")