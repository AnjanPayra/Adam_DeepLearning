"""
Step 5: Adam-Optimizer NN + Quantum Computing model
Adds two more models to the comparison:

  (A) "Adam-Optimized DNN"   - plain feed-forward net, trained purely with
      the Adam optimizer (no dropout/batchnorm/L2), to isolate what Adam
      itself buys vs the heavily-regularized Advanced Deep NN.

  (B) "Quantum VQC"          - a Variational Quantum Classifier: input
      features are PCA-reduced to n_qubits dimensions, angle-encoded onto
      qubits, run through a parameterized entangling circuit (simulated
      with a from-scratch NumPy statevector simulator - no qiskit/pennylane
      available in this sandbox/no internet), and trained with Adam via the
      exact parameter-shift rule for gradients.

Both are tuned with a small hyperparameter search on the validation split,
using the exact same train/val/test partition as the earlier basic &
advanced models for a fair comparison.
"""
import sys
sys.path.insert(0, '/home/claude/work')
import numpy as np
import pandas as pd
import itertools, random, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from adam_plain_nn import AdamPlainNN
from quantum_vqc import VQC

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ------------------------------------------------------------------
# Load the SAME splits used for the basic/advanced models
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

# ====================================================================
# (A) Adam-Optimized plain DNN -- hyperparameter search
# ====================================================================
print("=" * 70)
print("(A) ADAM-OPTIMIZED DNN -- hyperparameter search")
print("=" * 70)

adam_search_space = {
    'hidden': [(32,), (64, 32), (128, 64)],
    'lr': [1e-3, 3e-3, 1e-2],
    'class_weight': [1.0, 2.0, 3.0],
}
keys = list(adam_search_space.keys())
combos = list(itertools.product(*adam_search_space.values()))
random.shuffle(combos)
trials = combos[:6]

adam_results = []
for i, combo in enumerate(trials):
    cfg = dict(zip(keys, combo))
    layer_sizes = [n_features, *cfg['hidden'], 1]
    model = AdamPlainNN(layer_sizes, lr=cfg['lr'], seed=RANDOM_STATE)
    val_auc = model.fit(X_train_s, y_train, X_val_s, y_val, epochs=120,
                         batch_size=64, class_weight=cfg['class_weight'], patience=15)
    adam_results.append({**cfg, 'val_auc': val_auc})
    print(f"trial {i+1:2d}/10  hidden={cfg['hidden']}  lr={cfg['lr']:.0e}  "
          f"cw={cfg['class_weight']}  -> val_auc={val_auc:.4f}")

adam_results_df = pd.DataFrame(adam_results).sort_values('val_auc', ascending=False)
best_adam_cfg = adam_results_df.iloc[0].to_dict()
print("\nBest Adam-DNN config:", best_adam_cfg)

layer_sizes = [n_features, *best_adam_cfg['hidden'], 1]
best_adam_model = AdamPlainNN(layer_sizes, lr=best_adam_cfg['lr'], seed=RANDOM_STATE)
best_adam_model.fit(X_train_s, y_train, X_val_s, y_val, epochs=300, batch_size=64,
                    class_weight=best_adam_cfg['class_weight'], patience=25, verbose=True)

# ====================================================================
# (B) Quantum VQC -- PCA reduce to n_qubits, then hyperparameter search
# ====================================================================
print("\n" + "=" * 70)
print("(B) QUANTUM VQC -- hyperparameter search")
print("=" * 70)

n_qubits = 4
pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE).fit(X_train_s)
print(f"PCA({n_qubits}) explained variance ratio: {pca.explained_variance_ratio_.round(3)} "
      f"(cumulative {pca.explained_variance_ratio_.sum():.3f})")

def to_angles(X):
    """Scale PCA components to a [-pi, pi] angle-encoding range."""
    Xp = pca.transform(X)
    Xp = Xp / (np.max(np.abs(Xp), axis=0) + 1e-8)
    return Xp * np.pi

Xq_train = to_angles(X_train_s)
Xq_val   = to_angles(X_val_s)
Xq_test  = to_angles(X_test_s)

quantum_search_space = {
    'depth': [1, 2],
    'lr': [0.05, 0.1],
    'class_weight': [1.0, 2.0],
}
keys_q = list(quantum_search_space.keys())
combos_q = list(itertools.product(*quantum_search_space.values()))
random.shuffle(combos_q)
trials_q = combos_q[:4]

# Use a random subsample of the training set for the search (statevector
# simulation is done sample-by-sample in pure NumPy, so keep the search cheap)
sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(Xq_train), size=300, replace=False)
Xq_train_sub, y_train_sub = Xq_train[sub_idx], y_train[sub_idx]

quantum_results = []
for i, combo in enumerate(trials_q):
    cfg = dict(zip(keys_q, combo))
    vqc = VQC(n_qubits=n_qubits, depth=cfg['depth'], lr=cfg['lr'], seed=RANDOM_STATE)
    val_auc = vqc.fit(Xq_train_sub, y_train_sub, Xq_val, y_val, epochs=6,
                       batch_size=16, class_weight=cfg['class_weight'], patience=4)
    quantum_results.append({**cfg, 'val_auc': val_auc})
    print(f"trial {i+1}/4  depth={cfg['depth']}  lr={cfg['lr']}  "
          f"cw={cfg['class_weight']}  -> val_auc={val_auc:.4f}")

quantum_results_df = pd.DataFrame(quantum_results).sort_values('val_auc', ascending=False)
best_q_cfg = quantum_results_df.iloc[0].to_dict()
print("\nBest Quantum VQC config:", best_q_cfg)

best_vqc = VQC(n_qubits=n_qubits, depth=int(best_q_cfg['depth']), lr=best_q_cfg['lr'],
               seed=RANDOM_STATE)
best_vqc.fit(Xq_train_sub, y_train_sub, Xq_val, y_val, epochs=15, batch_size=16,
             class_weight=best_q_cfg['class_weight'], patience=6, verbose=True)

# ------------------------------------------------------------------
# Save everything for the unified validation/comparison step
# ------------------------------------------------------------------
with open('/home/claude/work/adam_plain_model.pkl', 'wb') as f:
    pickle.dump({'model': best_adam_model, 'best_cfg': best_adam_cfg}, f)
with open('/home/claude/work/quantum_vqc_model.pkl', 'wb') as f:
    pickle.dump({'model': best_vqc, 'best_cfg': best_q_cfg, 'pca': pca}, f)
adam_results_df.to_csv('/home/claude/work/adam_search_results.csv', index=False)
quantum_results_df.to_csv('/home/claude/work/quantum_search_results.csv', index=False)

print("\nSaved adam_plain_model.pkl, quantum_vqc_model.pkl, and search result CSVs")
