"""
Step 2: BASIC Deep Learning Model
A single-hidden-layer Multi-Layer Perceptron (MLP) classifier, tuned with
GridSearchCV over a small hyperparameter grid.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
import joblib

RANDOM_STATE = 42

data = pd.read_csv('/home/claude/work/protein_features.csv')
feature_cols = [c for c in data.columns if c not in ('protein', 'label')]
X = data[feature_cols].values
y = data['label'].values

# 70% train / 15% validation(dev) / 15% held-out test  (stratified on label)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ----------------------------------------------------------------
# Hyperparameter tuning (basic model): 1 hidden layer
# ----------------------------------------------------------------
param_grid = {
    'hidden_layer_sizes': [(16,), (32,), (64,)],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [0.001, 0.01],
}

base_mlp = MLPClassifier(
    activation='relu', solver='adam', max_iter=500,
    early_stopping=True, n_iter_no_change=15,
    random_state=RANDOM_STATE
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(base_mlp, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
grid.fit(X_train_s, y_train)

print("\n=== BASIC MODEL: Grid Search Results ===")
print("Best params:", grid.best_params_)
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

best_basic = grid.best_estimator_

# Refit best model on train+val to fix its early-stopping split behaviour,
# then evaluate on the untouched test set
best_basic.fit(X_train_s, y_train)

def evaluate(model, X, y, name):
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred),
        'recall': recall_score(y, pred),
        'f1': f1_score(y, pred),
        'roc_auc': roc_auc_score(y, proba),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:>10s}: {v:.4f}")
    print(confusion_matrix(y, pred))
    return metrics

print("\n=== BASIC MODEL PERFORMANCE ===")
val_metrics_basic  = evaluate(best_basic, X_val_s, y_val, "Validation set")
test_metrics_basic = evaluate(best_basic, X_test_s, y_test, "Held-out test set")

print("\nFull classification report (test set):")
print(classification_report(y_test, best_basic.predict(X_test_s),
                             target_names=['non-essential', 'essential']))

# persist artefacts for later steps (advanced model comparison + validation script)
joblib.dump(scaler, '/home/claude/work/scaler.joblib')
joblib.dump(best_basic, '/home/claude/work/basic_model.joblib')
np.savez('/home/claude/work/splits.npz',
         X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)

print("\nSaved scaler.joblib, basic_model.joblib, splits.npz")
