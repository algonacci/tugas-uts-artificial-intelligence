import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import shap
import os
import sys

# Ensure result directory exists
if not os.path.exists('result'):
    os.makedirs('result')

# Logger class to write to both stdout and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to file + terminal
sys.stdout = Logger("result/rf_normal_output.txt")

# 1. Load Data
train_df = pd.read_csv('kdd_train.csv')
test_df = pd.read_csv('kdd_test.csv')

# 2. Preprocessing
categorical_cols = ['protocol_type', 'service', 'flag']

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Encode labels
le_label = LabelEncoder()
train_df['labels'] = le_label.fit_transform(train_df['labels'])
test_df['labels'] = test_df['labels'].map(lambda s: le_label.transform([s])[0] if s in le_label.classes_ else -1)

# 3. Feature Selection based on IJCS Paper (Bat Algorithm Results)
# Based on the paper, these 25 features gave the best accuracy (99.4%)
selected_features = [
    'duration', 'src_bytes', 'wrong_fragment', 'hot', 'num_failed_logins', 
    'num_access_files', 'count', 'srv_count', 'srv_serror_rate', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_serror_rate', 'protocol_type', 'service', 
    'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login', 
    'dst_host_srv_rerror_rate', 'dst_host_srv_serror_rate', 'srv_diff_host_rate', 
    'dst_host_srv_diff_host_rate', 'diff_srv_rate', 'same_srv_rate'
]

print(f"Selecting {len(selected_features)} features based on Bat Algorithm (Paper IJCS)...")

# Filter columns
X_train = train_df[selected_features]
y_train = train_df['labels']
X_test = test_df[selected_features]
y_test = test_df['labels']

import time

# 4. Training Random Forest
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Calculate Baseline CV Score for comparison with GA
print("Calculating Baseline CV F1-Macro (this might take a moment)...")

# Filter rare classes just like in GA to make apple-to-apple comparison
class_counts = y_train.value_counts()
common_classes = class_counts[class_counts >= 5].index
mask = y_train.isin(common_classes)
X_train_sub = X_train[mask]
y_train_sub = y_train[mask]

cv_scores = cross_val_score(model, X_train_sub, y_train_sub, cv=3, scoring='f1_macro', n_jobs=-1)
print(f"Baseline CV F1-Macro (Filtered Rare Classes): {cv_scores.mean():.4f}")

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"Training Time: {end_time - start_time:.4f} seconds")

# 5. Evaluation
y_pred = model.predict(X_test)

# Filter out -1 (Unknown labels) for fair evaluation
mask_known = y_test != -1
y_test_known = y_test[mask_known]
y_pred_known = y_pred[mask_known]

acc_known = accuracy_score(y_test_known, y_pred_known)
print(f"\nFinal Test Accuracy (Normal RF - Known Classes Only): {acc_known:.4f}")
print(f"(Excluded {len(y_test) - len(y_test_known)} samples with unseen labels)")

print("Classification Report (Known Classes Only):")
print(classification_report(y_test_known, y_pred_known, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test_known, y_pred_known)
print("\nConfusion Matrix:")
print(cm)

# Ensure result directory exists
import os
if not os.path.exists('result'):
    os.makedirs('result')

# Visualisasi Confusion Matrix
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_label.classes_,
            yticklabels=le_label.classes_,
            linewidths=0.5, linecolor='gray')
plt.title('Confusion Matrix (RF Normal)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('result/rf_normal_confusion_matrix.png', dpi=300)
plt.close()
print("Saved result/rf_normal_confusion_matrix.png")

# 6. Feature Importance (Custom Pretty Plot)
plt.figure(figsize=(12, 10))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
features_top = [selected_features[i] for i in indices]
importances_top = importances[indices]

ax = sns.barplot(x=importances_top, y=features_top, hue=features_top, palette="viridis", legend=False)
for i in ax.containers:
    ax.bar_label(i, fmt='%.3f', padding=3)

plt.title("Top 20 Feature Importance (RF Normal)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig('result/rf_normal_feature_importance.png', dpi=300)
plt.close()
print("Saved result/rf_normal_feature_importance.png")

# 7. SHAP Analysis
print("\nCalculating SHAP values...")
# TreeExplainer works for Random Forest
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for a subset of test data
sample_size = min(1000, len(X_test))
X_sample = X_test.iloc[:sample_size]
# check_additivity=False to avoid errors with some RF approximations
shap_values = explainer.shap_values(X_sample, check_additivity=False)

# Handle multi-class
if isinstance(shap_values, list):
    shap_values_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    shap_values_avg = np.abs(shap_values)

print(f"SHAP values calculated for {sample_size} samples")

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_avg, X_sample, max_display=20, show=False)
plt.title("SHAP Feature Importance (RF) - Paper Features")
plt.tight_layout()
plt.savefig('result/rf_normal_shap_summary.png')
plt.close()
print("Saved result/rf_normal_shap_summary.png")

# SHAP Bar Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_avg, X_sample, plot_type="bar", max_display=20, show=False)
plt.title("SHAP Feature Importance (Bar Plot - RF Normal)")
plt.tight_layout()
plt.savefig('result/rf_normal_shap_bar.png')
plt.close()
print("Saved result/rf_normal_shap_bar.png")

print("\nSHAP Analysis Complete!")
