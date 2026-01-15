import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import shap

# 1. Load Data
train_df = pd.read_csv('kdd_train.csv')
test_df = pd.read_csv('kdd_test.csv')

# 2. Preprocessing
categorical_cols = ['protocol_type', 'service', 'flag']

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined to avoid unseen labels in test
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Encode labels
le_label = LabelEncoder()
train_df['labels'] = le_label.fit_transform(train_df['labels'])
# Handle unseen labels in test if any (transform safely)
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

# 4. Training
print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    tree_method='hist' # Faster
)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in le_label.classes_], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Ensure result directory exists
import os
if not os.path.exists('result'):
    os.makedirs('result')

# Visualisasi Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_label.classes_,
            yticklabels=le_label.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('result/xgb_normal_confusion_matrix.png')
plt.close()
print("Saved result/xgb_normal_confusion_matrix.png")

# 6. Feature Importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=20)
plt.title("Top 20 Feature Importance")
plt.savefig('result/xgb_normal_feature_importance.png')
plt.close()
print("Saved result/xgb_normal_feature_importance.png")

# 7. SHAP Analysis
print("\nCalculating SHAP values...")
# Create SHAP explainer (TreeExplainer is optimized for tree models)
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for a subset of test data (biar ga terlalu berat)
# Take 1000 samples for SHAP analysis
sample_size = min(1000, len(X_test))
X_sample = X_test.iloc[:sample_size]
shap_values = explainer.shap_values(X_sample)

# If multi-class, shap_values is a list of arrays
if isinstance(shap_values, list):
    # For multi-class, analyze the first class or average across all classes
    shap_values_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    shap_values_avg = np.abs(shap_values)

print(f"SHAP values calculated for {sample_size} samples")

# SHAP Summary Plot (Feature Importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_avg, X_sample, max_display=20, show=False)
plt.title("SHAP Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig('result/xgb_normal_shap_summary.png')
plt.close()
print("Saved result/xgb_normal_shap_summary.png")

# SHAP Bar Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_avg, X_sample, plot_type="bar", max_display=20, show=False)
plt.title("SHAP Feature Importance (Bar Plot)")
plt.tight_layout()
plt.savefig('result/xgb_normal_shap_bar.png')
plt.close()
print("Saved result/xgb_normal_shap_bar.png")

print("\nSHAP Analysis Complete!")
print(f"Analyzed {sample_size} samples from test set")