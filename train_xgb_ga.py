import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import random
import time
import os
import shap
import matplotlib.pyplot as plt
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
sys.stdout = Logger("result/xgb_ga_output.txt")

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("Loading data...")
train_df = pd.read_csv('kdd_train.csv')
test_df = pd.read_csv('kdd_test.csv')

# Preprocessing
categorical_cols = ['protocol_type', 'service', 'flag']

for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined to avoid unseen labels in test
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

le_label = LabelEncoder()
train_df['labels'] = le_label.fit_transform(train_df['labels'])
test_df['labels'] = test_df['labels'].map(lambda s: le_label.transform([s])[0] if s in le_label.classes_ else -1)

# FIXED FEATURES from Paper (Bat Algorithm)
selected_features = [
    'duration', 'src_bytes', 'wrong_fragment', 'hot', 'num_failed_logins', 
    'num_access_files', 'count', 'srv_count', 'srv_serror_rate', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_serror_rate', 'protocol_type', 'service', 
    'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login', 
    'dst_host_srv_rerror_rate', 'dst_host_srv_serror_rate', 'srv_diff_host_rate', 
    'dst_host_srv_diff_host_rate', 'diff_srv_rate', 'same_srv_rate'
]

print(f"Using {len(selected_features)} Fixed Features from Paper for Tuning...")

X = train_df[selected_features]
y = train_df['labels']
X_test_final = test_df[selected_features]
y_test_final = test_df['labels']

# Split for GA Tuning
X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Increased subsample for better representation
sample_size = 20000 
X_train_sub = X_train_full.iloc[:sample_size]
y_train_sub = y_train_full.iloc[:sample_size]

# Filter out rare classes for tuning stability (need at least n_splits samples)
class_counts = y_train_sub.value_counts()
valid_classes = class_counts[class_counts >= 5].index
mask = y_train_sub.isin(valid_classes)
X_train_sub = X_train_sub[mask]
y_train_sub = y_train_sub[mask]

# Re-encode labels for the subset to ensure they are 0..K-1 contiguous for XGBoost
le_sub = LabelEncoder()
y_train_sub = le_sub.fit_transform(y_train_sub)

# ==========================================
# 2. IMPLEMENTASI GENETIC ALG. (HYPERPARAM TUNING)
# ==========================================
class FastGA_Tuning:
    def __init__(self, pop_size=20, n_generations=15, mutation_rate=0.2, crossover_rate=0.8):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Define Search Space (Expanded)
        self.param_bounds = {
            'learning_rate': (0.01, 0.3),    # Float
            'max_depth': (3, 15),            # Int
            'n_estimators': (50, 300),       # Int
            'subsample': (0.5, 1.0),         # Float
            'colsample_bytree': (0.5, 1.0),  # Float
            'gamma': (0, 5),                 # Float
            'min_child_weight': (1, 10)      # Int
        }
        
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = 0.0

    def random_gene(self):
        return {
            'learning_rate': random.uniform(*self.param_bounds['learning_rate']),
            'max_depth': random.randint(*self.param_bounds['max_depth']),
            'n_estimators': random.randint(*self.param_bounds['n_estimators']),
            'subsample': random.uniform(*self.param_bounds['subsample']),
            'colsample_bytree': random.uniform(*self.param_bounds['colsample_bytree']),
            'gamma': random.uniform(*self.param_bounds['gamma']),
            'min_child_weight': random.randint(*self.param_bounds['min_child_weight'])
        }

    def initialize_population(self):
        return [self.random_gene() for _ in range(self.pop_size)]

    def fitness(self, individual):
        # Use Scikit-Learn Wrapper for Cross-Validation flexibility
        model = xgb.XGBClassifier(
            n_estimators=int(individual['n_estimators']),
            learning_rate=individual['learning_rate'],
            max_depth=int(individual['max_depth']),
            subsample=individual['subsample'],
            colsample_bytree=individual['colsample_bytree'],
            gamma=individual['gamma'],
            min_child_weight=int(individual['min_child_weight']),
            tree_method='hist',
            n_jobs=-1,
            verbosity=0
        )
        
        # Use Cross-Validation (3-fold) with F1 Macro for robustness
        scores = cross_val_score(model, X_train_sub, y_train_sub, cv=3, scoring='f1_macro', n_jobs=-1)
        return scores.mean()

    def crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            # Uniform Crossover: Mix params from parent 1 and 2
            child = {}
            for key in p1.keys():
                child[key] = p1[key] if random.random() < 0.5 else p2[key]
            return child
        return p1.copy()

    def mutation(self, individual):
        # Mutate one random param
        if random.random() < self.mutation_rate:
            key = random.choice(list(self.param_bounds.keys()))
            bounds = self.param_bounds[key]
            if isinstance(bounds[0], int):
                val = random.randint(*bounds)
            else:
                val = random.uniform(*bounds)
                
            # Ensure int for specific params (though sklearn wrapper handles floats mostly, good practice)
            if key in ['max_depth', 'n_estimators', 'min_child_weight']:
                 individual[key] = int(val)
            else:
                 individual[key] = val
                 
        return individual

    def run(self):
        print("Starting GA Hyperparameter Tuning (XGB)...")
        start_time = time.time()
        
        for gen in range(self.n_generations):
            gen_start_time = time.time()
            fitnesses = [self.fitness(ind) for ind in self.population]
            max_fit = max(fitnesses)
            idx_best = fitnesses.index(max_fit)
            
            gen_duration = time.time() - gen_start_time
            
            if max_fit > self.best_fitness:
                self.best_fitness = max_fit
                self.best_solution = self.population[idx_best].copy()
                print(f"Gen {gen+1}: New Best CV F1-Macro = {max_fit:.4f} | Time: {gen_duration:.2f}s | Params: {self.best_solution}")
            else:
                print(f"Gen {gen+1}: Best CV F1-Macro = {self.best_fitness:.4f} | Time: {gen_duration:.2f}s")

            # Next Gen construction
            next_pop = [self.best_solution.copy()] # Elitism
            
            # Selection (Tournament)
            def tournament():
                a, b = random.sample(range(self.pop_size), 2)
                return self.population[a] if fitnesses[a] > fitnesses[b] else self.population[b]
            
            while len(next_pop) < self.pop_size:
                p1 = tournament()
                p2 = tournament()
                child = self.crossover(p1, p2)
                child = self.mutation(child)
                next_pop.append(child)
                
            self.population = next_pop

        print(f"GA Tuning Finished in {time.time() - start_time:.2f}s")
        return self.best_solution

# ==========================================
# 3. RUNNING TUNING
# ==========================================
# Increased population and generations
ga = FastGA_Tuning(pop_size=20, n_generations=15)
best_params = ga.run()

print("\nBest Parameters Found:", best_params)

# ==========================================
# 4. FINAL TRAINING
# ==========================================
print("\nTraining Final Model with Best Parameters...")

# Use Scikit-Learn Wrapper for easier evaluation/SHAP
final_model = xgb.XGBClassifier(
    n_estimators=int(best_params['n_estimators']),
    learning_rate=best_params['learning_rate'],
    max_depth=int(best_params['max_depth']),
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    min_child_weight=int(best_params['min_child_weight']),
    tree_method='hist',
    n_jobs=-1
)

final_model.fit(X, y)

# Evaluation
y_pred = final_model.predict(X_test_final)

# Filter out -1 (Unknown labels) for fair evaluation
mask_known = y_test_final != -1
y_test_known = y_test_final[mask_known]
y_pred_known = y_pred[mask_known]

acc_known = accuracy_score(y_test_known, y_pred_known)
print(f"\nFinal Test Accuracy (Known Classes Only): {acc_known:.4f}")
print(f"(Excluded {len(y_test_final) - len(y_test_known)} samples with unseen labels)")

print("Classification Report (Known Classes Only):")
print(classification_report(y_test_known, y_pred_known, zero_division=0))

# 4.1 Confusion Matrix (Known Classes)
cm = confusion_matrix(y_test_known, y_pred_known)
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_label.classes_, 
            yticklabels=le_label.classes_,
            linewidths=0.5, linecolor='gray')
plt.title('Confusion Matrix (XGB GA - Known Classes)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('result/xgb_ga_confusion_matrix.png', dpi=300)
plt.close()
print("Saved result/xgb_ga_confusion_matrix.png")

# 4.2 Feature Importance (Custom Pretty Plot)
plt.figure(figsize=(12, 10))
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
features_top = [selected_features[i] for i in indices]
importances_top = importances[indices]

ax = sns.barplot(x=importances_top, y=features_top, hue=features_top, palette="viridis", legend=False)
for i in ax.containers:
    ax.bar_label(i, fmt='%.3f', padding=3)

plt.title("Top 20 Feature Importance (XGB GA)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig('result/xgb_ga_feature_importance.png', dpi=300)
plt.close()
print("Saved result/xgb_ga_feature_importance.png")


# ==========================================
# 5. SHAP ANALYSIS
# ==========================================
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(final_model)

# Calculate SHAP values for a subset
sample_size = min(1000, len(X_test_final))
X_sample = X_test_final.iloc[:sample_size]
shap_values = explainer.shap_values(X_sample)

# Handle multi-class
if isinstance(shap_values, list):
    shap_values_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    shap_values_avg = np.abs(shap_values)

print(f"SHAP values calculated for {sample_size} samples")

# SHAP Summary Plot
plt.figure(figsize=(10, 8))
# Set max_display safely
shap.summary_plot(shap_values_avg, X_sample, max_display=min(20, X_sample.shape[1]), show=False)
plt.title("SHAP Feature Importance (XGB Tuned by GA)")
plt.tight_layout()
plt.savefig('result/xgb_ga_shap_summary.png')
plt.close()
print("Saved result/xgb_ga_shap_summary.png")

# SHAP Bar Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_avg, X_sample, plot_type="bar", max_display=min(20, X_sample.shape[1]), show=False)
plt.title("SHAP Feature Importance (Bar Plot - XGB Tuned)")
plt.tight_layout()
plt.savefig('result/xgb_ga_shap_bar.png')
plt.close()
print("Saved result/xgb_ga_shap_bar.png")

print("Analysis Complete!")
