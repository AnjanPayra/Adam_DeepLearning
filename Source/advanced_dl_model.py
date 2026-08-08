"""
Step 3: ADVANCED Deep Learning Model
A deep (multi-hidden-layer) neural network implemented from scratch in NumPy:
  - He-initialised weights
  - ReLU hidden activations, sigmoid output
  - Batch normalization
  - Dropout regularization
  - L2 weight decay
  - Adam optimizer with mini-batches
  - Class-weighted binary cross-entropy (handles the essential / non-essential
    class imbalance, ~24% positive)
Hyperparameters (architecture, learning rate, dropout, L2, class weight) are
tuned with a randomized search validated on a held-out validation split.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
import json, itertools, random
from deep_nn import DeepNN

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ------------------------------------------------------------------
# Load the same splits used for the basic model (fair comparison)
# ------------------------------------------------------------------
splits = np.load('/home/claude/work/splits.npz')
X_train, y_train = splits['X_train'], splits['y_train']
X_val, y_val     = splits['X_val'], splits['y_val']
X_test, y_test   = splits['X_test'], splits['y_test']

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

n_features = X_train_s.shape[1]

# ------------------------------------------------------------------
# Deep MLP implementation
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Hyperparameter random search
# ------------------------------------------------------------------
search_space = {
    'hidden': [(64, 32), (128, 64), (64, 32, 16)],
    'dropout': [0.1, 0.2, 0.3],
    'l2': [1e-5, 1e-4, 1e-3],
    'lr': [1e-3, 3e-3],
    'class_weight': [1.0, 2.0, 3.0],  # up-weight the minority "essential" class
}

n_trials = 12
keys = list(search_space.keys())
combos = list(itertools.product(*search_space.values()))
random.shuffle(combos)
trials = combos[:n_trials]

results = []
print(f"Running randomized hyperparameter search: {n_trials} trials\n")
for i, combo in enumerate(trials):
    cfg = dict(zip(keys, combo))
    layer_sizes = [n_features, *cfg['hidden'], 1]
    model = DeepNN(layer_sizes, dropout=cfg['dropout'], l2=cfg['l2'],
                   lr=cfg['lr'], class_weight=cfg['class_weight'])
    model.fit(X_train_s, y_train, X_val_s, y_val, epochs=120, batch_size=64,
              patience=15, verbose=False)
    val_proba = model.predict_proba(X_val_s)
    val_auc = roc_auc_score(y_val, val_proba)
    results.append({**cfg, 'val_auc': val_auc})
    print(f"trial {i+1:2d}/{n_trials}  hidden={cfg['hidden']}  dropout={cfg['dropout']}  "
          f"l2={cfg['l2']:.0e}  lr={cfg['lr']:.0e}  cw={cfg['class_weight']}  "
          f"-> val_auc={val_auc:.4f}")

results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)
print("\n=== ADVANCED MODEL: Top 5 hyperparameter configs ===")
print(results_df.head(5).to_string(index=False))

best_cfg = results_df.iloc[0].to_dict()
print("\nBest config:", best_cfg)

# ------------------------------------------------------------------
# Retrain best config for longer, then evaluate
# ------------------------------------------------------------------
layer_sizes = [n_features, *best_cfg['hidden'], 1]
best_advanced = DeepNN(layer_sizes, dropout=best_cfg['dropout'], l2=best_cfg['l2'],
                        lr=best_cfg['lr'], class_weight=best_cfg['class_weight'])
history = best_advanced.fit(X_train_s, y_train, X_val_s, y_val,
                              epochs=300, batch_size=64, patience=25, verbose=True)

def evaluate(model, X, y, name, threshold=0.5):
    proba = model.predict_proba(X)
    pred = (proba >= threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred),
        'recall': recall_score(y, pred),
        'f1': f1_score(y, pred),
        'roc_auc': roc_auc_score(y, proba),
    }
    print(f"\n--- {name} (threshold={threshold}) ---")
    for k, v in metrics.items():
        print(f"{k:>10s}: {v:.4f}")
    print(confusion_matrix(y, pred))
    return metrics

print("\n=== ADVANCED MODEL PERFORMANCE (default 0.5 threshold) ===")
val_metrics_adv  = evaluate(best_advanced, X_val_s, y_val, "Validation set")
test_metrics_adv = evaluate(best_advanced, X_test_s, y_test, "Held-out test set")

print("\nFull classification report (test set):")
print(classification_report(y_test, best_advanced.predict(X_test_s),
                             target_names=['non-essential', 'essential']))

# Save everything needed for later validation against the 3rd file
import pickle
with open('/home/claude/work/advanced_model.pkl', 'wb') as f:
    pickle.dump({'model': best_advanced, 'best_cfg': best_cfg}, f)
results_df.to_csv('/home/claude/work/hyperparam_search_results.csv', index=False)
print("\nSaved advanced_model.pkl and hyperparam_search_results.csv")
