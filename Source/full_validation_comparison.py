"""
Step 6: Full Validation & Comparison (4 models)
Basic MLP | Advanced Deep NN | Adam-Optimized DNN | Quantum VQC (simulated)

Evaluates all four on the SAME held-out test set (drawn from the
Essential_and_non_essential_dataset.xls ground truth), reporting metrics at
the default 0.5 threshold and at a validation-tuned (Youden's J) threshold,
and produces ROC / Precision-Recall plots for all four.
"""
import sys
sys.path.insert(0, '/home/claude/work')
import numpy as np
import pandas as pd
import joblib, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, matthews_corrcoef)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deep_nn import DeepNN            # needed to unpickle advanced_model.pkl
from adam_plain_nn import AdamPlainNN  # needed to unpickle adam_plain_model.pkl
from quantum_vqc import VQC            # needed to unpickle quantum_vqc_model.pkl

RANDOM_STATE = 42

# ---- load data splits ----
splits = np.load('/home/claude/work/splits.npz')
X_train, y_train = splits['X_train'], splits['y_train']
X_val, y_val     = splits['X_val'], splits['y_val']
X_test, y_test   = splits['X_test'], splits['y_test']

scaler = joblib.load('/home/claude/work/scaler.joblib')
X_val_s, X_test_s = scaler.transform(X_val), scaler.transform(X_test)

# ---- load all four models ----
basic_model = joblib.load('/home/claude/work/basic_model.joblib')

with open('/home/claude/work/advanced_model.pkl', 'rb') as f:
    advanced_model = pickle.load(f)['model']

with open('/home/claude/work/adam_plain_model.pkl', 'rb') as f:
    adam_model = pickle.load(f)['model']

with open('/home/claude/work/quantum_vqc_model.pkl', 'rb') as f:
    q_bundle = pickle.load(f)
    vqc_model = q_bundle['model']
    pca = q_bundle['pca']

def to_angles(X):
    Xp = pca.transform(X)
    Xp = Xp / (np.max(np.abs(Xp), axis=0) + 1e-8)
    return Xp * np.pi

Xq_val_s, Xq_test_s = to_angles(X_val_s), to_angles(X_test_s)

def best_threshold(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    return thr[np.argmax(tpr - fpr)]

def full_eval(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, pred),
        'roc_auc': roc_auc_score(y_true, proba),
    }, confusion_matrix(y_true, pred)

model_registry = [
    ('Basic MLP',            basic_model,    X_val_s,   X_test_s),
    ('Advanced Deep NN',     advanced_model, X_val_s,   X_test_s),
    ('Adam-Optimized DNN',   adam_model,     X_val_s,   X_test_s),
    ('Quantum VQC',          vqc_model,      Xq_val_s,  Xq_test_s),
]

results = {}
plt.figure(figsize=(12, 5))

for name, model, Xv, Xt in model_registry:
    val_proba = model.predict_proba(Xv)
    if val_proba.ndim == 2:
        val_proba = val_proba[:, 1]
    thr = best_threshold(y_val, val_proba)

    test_proba = model.predict_proba(Xt)
    if test_proba.ndim == 2:
        test_proba = test_proba[:, 1]

    metrics_default, cm_default = full_eval(y_test, test_proba, 0.5)
    metrics_tuned, cm_tuned = full_eval(y_test, test_proba, thr)
    results[name] = {'default': metrics_default, 'tuned': metrics_tuned,
                      'cm_default': cm_default, 'cm_tuned': cm_tuned, 'thr': thr}

    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"{name} (AUC={metrics_default['roc_auc']:.3f})")

    prec, rec, _ = precision_recall_curve(y_test, test_proba)
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, label=name)

    print(f"\n================ {name} ================")
    print(f"Optimal (Youden's J) threshold from validation set: {thr:.3f}")
    print("\n-- Test set @ default threshold 0.5 --")
    for k, v in metrics_default.items():
        print(f"  {k:>10s}: {round(v,4) if isinstance(v,float) else v}")
    print("Confusion matrix:\n", cm_default)
    print(f"\n-- Test set @ tuned threshold {thr:.3f} --")
    for k, v in metrics_tuned.items():
        print(f"  {k:>10s}: {round(v,4) if isinstance(v,float) else v}")
    print("Confusion matrix:\n", cm_tuned)

plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Test Set (4 models)'); plt.legend(fontsize=8)

plt.subplot(1, 2, 2)
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Precision-Recall Curve — Test Set (4 models)'); plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/work/model_comparison_4models.png', dpi=150)
print("\nSaved plot -> /home/claude/work/model_comparison_4models.png")

# ---- final summary table ----
rows = []
for name, res in results.items():
    for setting in ['default', 'tuned']:
        m = res[setting]
        label = 'default_threshold(0.5)' if setting == 'default' else f"tuned_threshold({res['thr']:.3f})"
        rows.append({'model': name, 'threshold_setting': label, **m})
summary = pd.DataFrame(rows)
summary.to_csv('/home/claude/work/final_comparison_table_4models.csv', index=False)
print("\n=== FINAL COMPARISON TABLE (4 MODELS) ===")
print(summary.round(4).to_string(index=False))
