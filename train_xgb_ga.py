import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import time
import os
import shap
import matplotlib.pyplot as plt

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
# Subsample for FAST Tuning
sample_size = 10000 
X_train_sub = X_train_full.iloc[:sample_size]
y_train_sub = y_train_full.iloc[:sample_size]
X_val_sub = X_val.iloc[:2000]
y_val_sub = y_val.iloc[:2000]

# ==========================================
# 2. IMPLEMENTASI GENETIC ALG. (HYPERPARAM TUNING)
# ==========================================
class FastGA_Tuning:
    def __init__(self, pop_size=10, n_generations=5, mutation_rate=0.1, crossover_rate=0.7):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Define Search Space
        self.param_bounds = {
            'learning_rate': (0.01, 0.3),    # Float
            'max_depth': (3, 10),            # Int
            'n_estimators': (50, 150),       # Int
            'subsample': (0.5, 1.0),         # Float
            'colsample_bytree': (0.5, 1.0),  # Float
            'gamma': (0, 5)                  # Float
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
            'gamma': random.uniform(*self.param_bounds['gamma'])
        }

    def initialize_population(self):
        return [self.random_gene() for _ in range(self.pop_size)]

    def fitness(self, individual):
        # Convert params to proper types
        params = {
            'objective': 'multi:softmax',
            'num_class': len(le_label.classes_),
            'tree_method': 'hist',
            'verbosity': 0,
            'nthread': 4,
            # Tuned Params
            'learning_rate': individual['learning_rate'],
            'max_depth': int(individual['max_depth']),
            'subsample': individual['subsample'],
            'colsample_bytree': individual['colsample_bytree'],
            'gamma': individual['gamma']
        }
        
        rounds = int(individual['n_estimators'])
        
        # Train on Subset
        dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
        dval = xgb.DMatrix(X_val_sub, label=y_val_sub)
        
        bst = xgb.train(params, dtrain, num_boost_round=rounds)
        preds = bst.predict(dval)
        acc = accuracy_score(y_val_sub, preds)
        
        return acc

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
            individual[key] = val
        return individual

    def run(self):
        print("Starting GA Hyperparameter Tuning (XGB)...")
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
ga = FastGA_Tuning(pop_size=10, n_generations=5)
best_params = ga.run()

print("\nBest Parameters Found:", best_params)

# ==========================================
# 4. FINAL TRAINING
# ==========================================
print("\nTraining Final Model with Best Parameters...")

final_params = {
    'objective': 'multi:softmax',
    'num_class': len(le_label.classes_),
    'tree_method': 'hist',
    'learning_rate': best_params['learning_rate'],
    'max_depth': int(best_params['max_depth']),
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'gamma': best_params['gamma'],
    'n_jobs': -1
}

# Use Scikit-Learn Wrapper for easier evaluation/SHAP
final_model = xgb.XGBClassifier(
    n_estimators=int(best_params['n_estimators']),
    learning_rate=best_params['learning_rate'],
    max_depth=int(best_params['max_depth']),
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    tree_method='hist',
    n_jobs=-1
)

final_model.fit(X, y)

# Evaluation
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
