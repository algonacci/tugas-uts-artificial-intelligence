import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import random
import time
import os
import shap
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
sys.stdout = Logger("result/rf_ga_output.txt")

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
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

le_label = LabelEncoder()
train_df['labels'] = le_label.fit_transform(train_df['labels'])
test_df['labels'] = test_df['labels'].map(lambda s: le_label.transform([s])[0] if s in le_label.classes_ else -1)

# FIXED FEATURES from Paper
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

# Split for Tuning
X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
sample_size = 20000  # Increased sample size for better stability
X_train_sub = X_train_full.iloc[:sample_size]
y_train_sub = y_train_full.iloc[:sample_size]

# Filter out rare classes for tuning stability
class_counts = y_train_sub.value_counts()
valid_classes = class_counts[class_counts >= 5].index
mask = y_train_sub.isin(valid_classes)
X_train_sub = X_train_sub[mask]
y_train_sub = y_train_sub[mask]

# ==========================================
# 2. IMPLEMENTASI GENETIC ALG. (RF TUNING)
# ==========================================
class FastGA_Tuning_RF:
    def __init__(self, pop_size=30, n_generations=20, mutation_rate=0.2, crossover_rate=0.8):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Define Search Space (Expanded)
        self.param_bounds = {
            'n_estimators': (50, 500),       # Int
            'max_depth': (5, 50),            # Int
            'min_samples_split': (2, 20),    # Int
            'min_samples_leaf': (1, 10),     # Int
            'max_features_idx': (0, 2)       # Int: 0=sqrt, 1=log2, 2=None
        }
        self.max_features_map = {0: 'sqrt', 1: 'log2', 2: None}
        
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = 0.0

    def random_gene(self):
        gene = {}
        for key, bounds in self.param_bounds.items():
            gene[key] = random.randint(*bounds)
        return gene

    def initialize_population(self):
        return [self.random_gene() for _ in range(self.pop_size)]

    def fitness(self, individual):
        max_feat = self.max_features_map[individual['max_features_idx']]
        
        model = RandomForestClassifier(
            n_estimators=individual['n_estimators'],
            max_depth=individual['max_depth'],
            min_samples_split=individual['min_samples_split'],
            min_samples_leaf=individual['min_samples_leaf'],
            max_features=max_feat,
            n_jobs=-1,
            random_state=42
        )
        
        # Use Cross-Validation (3-fold) with F1 Maintained (Macro) for imbalance handling directly in loop or here
        # Using accuracy as requested but on CV for robustness
        # Switching to f1_macro for better handling of minority classes
        scores = cross_val_score(model, X_train_sub, y_train_sub, cv=3, scoring='f1_macro', n_jobs=-1, error_score='raise')
        return scores.mean()

    def crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            child = {}
            for key in p1.keys():
                child[key] = p1[key] if random.random() < 0.5 else p2[key]
            return child
        return p1.copy()

    def mutation(self, individual):
        if random.random() < self.mutation_rate:
            key = random.choice(list(self.param_bounds.keys()))
            bounds = self.param_bounds[key]
            val = random.randint(*bounds)
            individual[key] = val
        return individual

    def run(self):
        print("Starting GA Hyperparameter Tuning (RF)...")
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

            next_pop = [self.best_solution.copy()]
            
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
# Increased population and generations for better exploration
ga = FastGA_Tuning_RF(pop_size=20, n_generations=15) 
best_params_raw = ga.run()

# Decode params
best_params = best_params_raw.copy()
best_params['max_features'] = ga.max_features_map[best_params_raw['max_features_idx']]
del best_params['max_features_idx']

print("\nBest Parameters Found:", best_params)

# ==========================================
# 4. FINAL TRAINING
# ==========================================
print("\nTraining Final RF Model with Best Parameters...")
final_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    n_jobs=-1,
    random_state=42
)

final_model.fit(X, y)

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
sns.set_style("whitegrid")  # Ensure style is set if reused
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_label.classes_, 
            yticklabels=le_label.classes_,
            linewidths=0.5, linecolor='gray')
plt.title('Confusion Matrix (RF GA - Known Classes)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('result/rf_ga_confusion_matrix.png', dpi=300)
plt.close()
print("Saved result/rf_ga_confusion_matrix.png")

# 4.2 Feature Importance (Built-in)
# Enhancing this to match the seaborn style of other scripts
plt.figure(figsize=(12, 10))
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1][:20]  # Top 20
features_top = [selected_features[i] for i in indices]
importances_top = importances[indices]

ax = sns.barplot(x=importances_top, y=features_top, hue=features_top, palette="viridis", legend=False)
for i in ax.containers:
    ax.bar_label(i, fmt='%.3f', padding=3)

plt.title("Top 20 Feature Importance (RF GA)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig('result/rf_ga_feature_importance.png', dpi=300)
plt.close()
print("Saved result/rf_ga_feature_importance.png")

# ==========================================
# 5. SHAP ANALYSIS
# ==========================================
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(final_model)
sample_size = min(1000, len(X_test_final))
X_sample = X_test_final.iloc[:sample_size]
shap_values = explainer.shap_values(X_sample, check_additivity=False)

if isinstance(shap_values, list):
    shap_values_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    shap_values_avg = np.abs(shap_values)

print(f"SHAP values calculated for {sample_size} samples")

# Save Plots
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_avg, X_sample, max_display=min(20, X_sample.shape[1]), show=False)
plt.title("SHAP Feature Importance (RF Tuned by GA)")
plt.tight_layout()
plt.savefig('result/rf_ga_shap_summary.png')
plt.close()
print("Saved result/rf_ga_shap_summary.png")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_avg, X_sample, plot_type="bar", max_display=min(20, X_sample.shape[1]), show=False)
plt.title("SHAP Feature Importance (Bar Plot - RF Tuned)")
plt.tight_layout()
plt.savefig('result/rf_ga_shap_bar.png')
plt.close()
print("Saved result/rf_ga_shap_bar.png")

print("Analysis Complete!")
