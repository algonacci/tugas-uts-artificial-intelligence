import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import random
import time
import os
import shap

# Ensure result directory exists
if not os.path.exists('result'):
    os.makedirs('result')

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
sample_size = 10000 
X_train_sub = X_train_full.iloc[:sample_size]
y_train_sub = y_train_full.iloc[:sample_size]
X_val_sub = X_val.iloc[:2000]
y_val_sub = y_val.iloc[:2000]

# ==========================================
# 2. IMPLEMENTASI GENETIC ALG. (RF TUNING)
# ==========================================
class FastGA_Tuning_RF:
    def __init__(self, pop_size=10, n_generations=5, mutation_rate=0.1, crossover_rate=0.7):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Define Search Space
        self.param_bounds = {
            'n_estimators': (50, 200),       # Int
            'max_depth': (5, 30),            # Int
            'min_samples_split': (2, 10),    # Int
            'min_samples_leaf': (1, 5),      # Int
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
        
        model.fit(X_train_sub, y_train_sub)
        preds = model.predict(X_val_sub)
        acc = accuracy_score(y_val_sub, preds)
        return acc

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
            fitnesses = [self.fitness(ind) for ind in self.population]
            max_fit = max(fitnesses)
            idx_best = fitnesses.index(max_fit)
            
            if max_fit > self.best_fitness:
                self.best_fitness = max_fit
                self.best_solution = self.population[idx_best].copy()
                print(f"Gen {gen+1}: New Best Acc = {max_fit:.4f} | Params: {self.best_solution}")
            else:
                print(f"Gen {gen+1}: Best Acc = {self.best_fitness:.4f}")

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
ga = FastGA_Tuning_RF(pop_size=10, n_generations=5)
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
acc = accuracy_score(y_test_final, y_pred)
print(f"\nFinal Test Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test_final, y_pred, zero_division=0))

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
