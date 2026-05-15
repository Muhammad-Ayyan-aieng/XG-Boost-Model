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
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

os.makedirs("outputs/figures", exist_ok=True)

print("=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

print("\n1. Loading data...")
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
print(f"   Test set size: {len(X_test):,} rows")

print("\n2. Loading models...")
default_model = joblib.load("outputs/models/final_model.pkl")
tuned_model = joblib.load("outputs/models/tuned_model.pkl")
print("   Models loaded")

print("\n3. Making predictions...")
y_pred_default = default_model.predict(X_test)
y_pred_tuned = tuned_model.predict(X_test)
default_acc = accuracy_score(y_test, y_pred_default)
tuned_acc = accuracy_score(y_test, y_pred_tuned)
print(f"   Default accuracy: {default_acc*100:.2f}%")
print(f"   Tuned accuracy: {tuned_acc*100:.2f}%")

# Calculate recall for each severity
print("\n   Recall by severity (Tuned Model):")
for i in range(4):
    recall = recall_score(y_test, y_pred_tuned, labels=[i], average=None)[0]
    print(f"   Severity {i+1}: {recall*100:.1f}%")

# =========================================================
# FIGURE 1: CONFUSION MATRIX (Tuned Model)
# =========================================================
print("\n4. Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_tuned)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'],
            yticklabels=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix - Tuned XGBoost', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrix.png', dpi=150)
plt.close()
print("   Saved: outputs/figures/confusion_matrix.png")

# =========================================================
# FIGURE 2: FEATURE IMPORTANCE (Tuned Model)
# =========================================================
print("\n5. Generating feature importance chart...")
importance = tuned_model.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
colors = plt.cm.RdYlGn_r(feat_imp['importance'] / feat_imp['importance'].max())
plt.barh(feat_imp['feature'], feat_imp['importance'], color=colors, edgecolor='black')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 15 Feature Importance - Tuned XGBoost', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/figures/feature_importance.png', dpi=150)
plt.close()
print("   Saved: outputs/figures/feature_importance.png")

# =========================================================
# FIGURE 3: ACCURACY COMPARISON (Default vs Tuned)
# =========================================================
print("\n6. Generating accuracy comparison chart...")
plt.figure(figsize=(6, 6))
bars = plt.bar(['Default Model', 'Tuned Model'], [default_acc*100, tuned_acc*100],
               color=['#3498db', '#2ecc71'], edgecolor='black', width=0.6)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Default vs Tuned XGBoost Performance', fontsize=14)
plt.ylim(60, 85)

for bar, acc in zip(bars, [default_acc*100, tuned_acc*100]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/figures/accuracy_comparison.png', dpi=150)
plt.close()
print("   Saved: outputs/figures/accuracy_comparison.png")

# =========================================================
# FIGURE 4: SEVERITY DISTRIBUTION IN DATASET
# =========================================================
print("\n7. Generating severity distribution chart...")
severity_counts = df['Severity'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
bars = plt.bar([1, 2, 3, 4], severity_counts.values, color=colors, edgecolor='black')
plt.xlabel('Severity Level', fontsize=12)
plt.ylabel('Number of Accidents', fontsize=12)
plt.title('Severity Distribution in Dataset', fontsize=14)
plt.xticks([1, 2, 3, 4])

for bar, count in zip(bars, severity_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'{count:,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figures/severity_distribution.png', dpi=150)
plt.close()
print("   Saved: outputs/figures/severity_distribution.png")

# =========================================================
# FIGURE 5: PER CLASS ACCURACY (Tuned Model)
# =========================================================
print("\n8. Generating per-class accuracy chart...")
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
per_class_acc = cm_normalized.diagonal() * 100

plt.figure(figsize=(8, 6))
bars = plt.bar([1, 2, 3, 4], per_class_acc, color=colors, edgecolor='black')
plt.xlabel('Severity Level', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy - Tuned XGBoost', fontsize=14)
plt.ylim(0, 100)
plt.xticks([1, 2, 3, 4])

for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figures/per_class_accuracy.png', dpi=150)
plt.close()
print("   Saved: outputs/figures/per_class_accuracy.png")

# =========================================================
# FIGURE 6: RECALL COMPARISON CHART (NEW)
# =========================================================
print("\n9. Generating recall comparison chart...")
recalls = []
for i in range(4):
    recall = recall_score(y_test, y_pred_tuned, labels=[i], average=None)[0]
    recalls.append(recall * 100)

plt.figure(figsize=(8, 6))
bars = plt.bar([1, 2, 3, 4], recalls, color=colors, edgecolor='black')
plt.xlabel('Severity Level', fontsize=12)
plt.ylabel('Recall (%)', fontsize=12)
plt.title('Recall by Severity - Tuned XGBoost', fontsize=14)
plt.ylim(0, 100)
plt.xticks([1, 2, 3, 4])

for bar, rec in zip(bars, recalls):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{rec:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/figures/recall_by_severity.png', dpi=150)
plt.close()
print("   Saved: outputs/figures/recall_by_severity.png")

# =========================================================
# FIGURE 7: PREDICTION DISTRIBUTION VS ACTUAL
# =========================================================
print("\n10. Generating prediction distribution chart...")
pred_dist = np.bincount(y_pred_tuned, minlength=4)
actual_dist = np.bincount(y_test, minlength=4)

x = np.arange(4)
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, actual_dist, width, label='Actual', color='#3498db', edgecolor='black')
plt.bar(x + width/2, pred_dist, width, label='Predicted', color='#e74c3c', edgecolor='black')
plt.xlabel('Severity Level', fontsize=12)
plt.ylabel('Number of Accidents', fontsize=12)
plt.title('Actual vs Predicted Distribution', fontsize=14)
plt.xticks(x, [1, 2, 3, 4])
plt.legend()

plt.tight_layout()
plt.savefig('outputs/figures/prediction_distribution.png', dpi=150)
plt.close()
print("   Saved: outputs/figures/prediction_distribution.png")

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print("\nOutput files saved in: outputs/figures/")
print("   - confusion_matrix.png")
print("   - feature_importance.png")
print("   - accuracy_comparison.png")
print("   - severity_distribution.png")
print("   - per_class_accuracy.png")
print("   - recall_by_severity.png")
print("   - prediction_distribution.png")
