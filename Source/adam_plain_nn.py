"""
adam_plain_nn.py
A plain multi-layer perceptron (He-init, ReLU, sigmoid output) trained
*purely* with the Adam optimizer and ordinary (class-weighted) binary
cross-entropy -- no dropout, no batch-norm, no L2. This isolates what the
Adam optimizer itself contributes, as a distinct comparison point from the
heavily-regularized "Advanced Deep NN" (which also uses Adam, but adds
dropout + batchnorm + L2 on top of it).
"""
import numpy as np
from sklearn.metrics import roc_auc_score


class AdamPlainNN:
    def __init__(self, layer_sizes, lr=1e-3, seed=42):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.L = len(layer_sizes) - 1
        rng = np.random.RandomState(seed)
        self.params = {}
        for l in range(1, self.L + 1):
            fan_in = layer_sizes[l - 1]
            self.params[f'W{l}'] = rng.randn(layer_sizes[l - 1], layer_sizes[l]) * np.sqrt(2.0 / fan_in)
            self.params[f'b{l}'] = np.zeros((1, layer_sizes[l]))
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def forward(self, X):
        cache = {'A0': X}
        A = X
        for l in range(1, self.L + 1):
            Z = A @ self.params[f'W{l}'] + self.params[f'b{l}']
            A = self.relu(Z) if l < self.L else self.sigmoid(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        return A, cache

    def backward(self, cache, y, class_weight):
        m = y.shape[0]
        y = y.reshape(-1, 1)
        w = np.where(y == 1, class_weight, 1.0)
        grads = {}
        dZ = w * (cache[f'A{self.L}'] - y) / m
        for l in range(self.L, 0, -1):
            A_prev = cache[f'A{l-1}']
            grads[f'W{l}'] = A_prev.T @ dZ
            grads[f'b{l}'] = dZ.sum(axis=0, keepdims=True)
            if l > 1:
                dA_prev = dZ @ self.params[f'W{l}'].T
                dZ = dA_prev * (cache[f'Z{l-1}'] > 0)
        return grads

    def adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for k in grads:
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, y, X_val=None, y_val=None, epochs=150, batch_size=64,
            class_weight=1.0, patience=15, verbose=False):
        n = X.shape[0]
        best_val_auc, best_params, no_improve = -1, None, 0
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            Xs, ys = X[perm], y[perm]
            for i in range(0, n, batch_size):
                xb, yb = Xs[i:i + batch_size], ys[i:i + batch_size]
                _, cache = self.forward(xb)
                grads = self.backward(cache, yb, class_weight)
                self.adam_step(grads)
            if X_val is not None:
                val_pred, _ = self.forward(X_val)
                val_auc = roc_auc_score(y_val, val_pred.ravel())
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_params = {k: v.copy() for k, v in self.params.items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if verbose and epoch % 20 == 0:
                    print(f"  [Adam-NN] epoch {epoch:3d}  val_auc={val_auc:.4f}")
                if no_improve >= patience:
                    break
        if best_params is not None:
            self.params = best_params
        return best_val_auc

    def predict_proba(self, X):
        p, _ = self.forward(X)
        return p.ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
