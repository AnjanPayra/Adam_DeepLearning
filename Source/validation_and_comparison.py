"""
import sys; sys.path.insert(0, '/home/claude/work')
Step 4: Validation & Model Comparison
Loads both tuned models (basic MLP, advanced deep NN), selects an optimal
decision threshold from the validation set (Youden's J statistic — important
here because essential proteins are the minority class, ~24%), then validates
both models on the untouched held-out test set drawn from the
Essential_and_non_essential_dataset.xls ground truth. Produces a comparison
table + ROC / PR plots.
"""
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, matthews_corrcoef)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# ---- load data splits + models ----
splits = np.load('/home/claude/work/splits.npz')
X_train, y_train = splits['X_train'], splits['y_train']
X_val, y_val     = splits['X_val'], splits['y_val']
X_test, y_test   = splits['X_test'], splits['y_test']

scaler = joblib.load('/home/claude/work/scaler.joblib')
X_val_s, X_test_s = scaler.transform(X_val), scaler.transform(X_test)

basic_model = joblib.load('/home/claude/work/basic_model.joblib')
with open('/home/claude/work/advanced_model.pkl', 'rb') as f:
    adv_bundle = pickle.load(f)
advanced_model = adv_bundle['model']

def best_threshold(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    return thr[np.argmax(j)]

def full_eval(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred),
        'recall': recall_score(y_true, pred),
        'f1': f1_score(y_true, pred),
        'mcc': matthews_corrcoef(y_true, pred),
        'roc_auc': roc_auc_score(y_true, proba),
    }, confusion_matrix(y_true, pred)

results = {}
plt.figure(figsize=(12, 5))

for name, model in [('Basic MLP', basic_model), ('Advanced Deep NN', advanced_model)]:
    val_proba = model.predict_proba(X_val_s)
    if val_proba.ndim == 2:
        val_proba = val_proba[:, 1]
    thr = best_threshold(y_val, val_proba)

    test_proba = model.predict_proba(X_test_s)
    if test_proba.ndim == 2:
        test_proba = test_proba[:, 1]

    metrics_default, cm_default = full_eval(y_test, test_proba, 0.5)
    metrics_tuned, cm_tuned = full_eval(y_test, test_proba, thr)

    results[name] = {
        'default_threshold(0.5)': metrics_default,
        f'tuned_threshold({thr:.3f})': metrics_tuned,
        'confusion_matrix_tuned': cm_tuned.tolist(),
    }

    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"{name} (AUC={metrics_default['roc_auc']:.3f})")

    prec, rec, _ = precision_recall_curve(y_test, test_proba)
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, label=name)

    print(f"\n================ {name} ================")
    print("Optimal (Youden's J) threshold from validation set:", round(thr, 3))
    print("\n-- Test set @ default threshold 0.5 --")
    for k, v in metrics_default.items():
        print(f"  {k:>10s}: {v if isinstance(v,str) else round(v,4)}")
    print("Confusion matrix:\n", cm_default)
    print(f"\n-- Test set @ tuned threshold {thr:.3f} --")
    for k, v in metrics_tuned.items():
        print(f"  {k:>10s}: {v if isinstance(v,str) else round(v,4)}")
    print("Confusion matrix:\n", cm_tuned)

plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Test Set'); plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Precision-Recall Curve — Test Set'); plt.legend()

plt.tight_layout()
plt.savefig('/home/claude/work/model_comparison.png', dpi=150)
print("\nSaved plot -> /home/claude/work/model_comparison.png")

# ---- summary comparison table ----
rows = []
for name, res in results.items():
    for thr_name in ['default_threshold(0.5)'] + [k for k in res if k.startswith('tuned')]:
        m = res[thr_name]
        rows.append({'model': name, 'threshold_setting': thr_name, **m})
summary = pd.DataFrame(rows)
summary.to_csv('/home/claude/work/final_comparison_table.csv', index=False)
print("\n=== FINAL COMPARISON TABLE ===")
print(summary.round(4).to_string(index=False))
