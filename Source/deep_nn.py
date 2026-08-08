import numpy as np
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

class DeepNN:
    def __init__(self, layer_sizes, dropout=0.2, l2=1e-4, lr=1e-3,
                 class_weight=1.0, seed=RANDOM_STATE):
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.l2 = l2
        self.lr = lr
        self.class_weight = class_weight  # weight applied to the positive (essential) class
        rng = np.random.RandomState(seed)
        self.params = {}
        self.bn = {}
        L = len(layer_sizes) - 1
        self.L = L
        for l in range(1, L + 1):
            fan_in = layer_sizes[l - 1]
            self.params[f'W{l}'] = rng.randn(layer_sizes[l - 1], layer_sizes[l]) * np.sqrt(2.0 / fan_in)
            self.params[f'b{l}'] = np.zeros((1, layer_sizes[l]))
            if l < L:  # batch-norm on hidden layers only
                self.bn[f'gamma{l}'] = np.ones((1, layer_sizes[l]))
                self.bn[f'beta{l}']  = np.zeros((1, layer_sizes[l]))
                self.bn[f'run_mean{l}'] = np.zeros((1, layer_sizes[l]))
                self.bn[f'run_var{l}']  = np.ones((1, layer_sizes[l]))
        # Adam moments
        self.m = {k: np.zeros_like(v) for k, v in {**self.params, **self.bn}.items()
                   if not k.startswith('run_')}
        self.v = {k: np.zeros_like(v) for k, v in {**self.params, **self.bn}.items()
                   if not k.startswith('run_')}
        self.t = 0

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def forward(self, X, training=True):
        cache = {'A0': X}
        A = X
        for l in range(1, self.L + 1):
            Z = A @ self.params[f'W{l}'] + self.params[f'b{l}']
            if l < self.L:
                # batch norm
                if training:
                    mu = Z.mean(axis=0, keepdims=True)
                    var = Z.var(axis=0, keepdims=True)
                    self.bn[f'run_mean{l}'] = 0.9 * self.bn[f'run_mean{l}'] + 0.1 * mu
                    self.bn[f'run_var{l}']  = 0.9 * self.bn[f'run_var{l}']  + 0.1 * var
                else:
                    mu, var = self.bn[f'run_mean{l}'], self.bn[f'run_var{l}']
                Zn = (Z - mu) / np.sqrt(var + 1e-8)
                Zbn = self.bn[f'gamma{l}'] * Zn + self.bn[f'beta{l}']
                A = self.relu(Zbn)
                if training and self.dropout > 0:
                    mask = (np.random.rand(*A.shape) > self.dropout) / (1 - self.dropout)
                    A = A * mask
                    cache[f'mask{l}'] = mask
                cache[f'Z{l}'] = Z; cache[f'Zn{l}'] = Zn; cache[f'mu{l}'] = mu; cache[f'var{l}'] = var
            else:
                A = self.sigmoid(Z)
                cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        return A, cache

    def compute_loss(self, y_hat, y):
        y = y.reshape(-1, 1)
        w = np.where(y == 1, self.class_weight, 1.0)
        eps = 1e-8
        bce = -(w * (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))
        data_loss = bce.mean()
        l2_loss = sum(np.sum(self.params[f'W{l}'] ** 2) for l in range(1, self.L + 1))
        return data_loss + self.l2 * l2_loss

    def backward(self, cache, y):
        m = y.shape[0]
        y = y.reshape(-1, 1)
        w = np.where(y == 1, self.class_weight, 1.0)
        grads = {}
        A_L = cache[f'A{self.L}']
        dZ = w * (A_L - y) / m
        for l in range(self.L, 0, -1):
            A_prev = cache[f'A{l-1}']
            grads[f'W{l}'] = A_prev.T @ dZ + 2 * self.l2 * self.params[f'W{l}']
            grads[f'b{l}'] = dZ.sum(axis=0, keepdims=True)
            if l > 1:
                dA_prev = dZ @ self.params[f'W{l}'].T
                lp = l - 1
                if self.dropout > 0 and f'mask{lp}' in cache:
                    dA_prev = dA_prev * cache[f'mask{lp}']
                dZbn = dA_prev * (cache[f'Zn{lp}'] * 0 + (cache[f'A{lp}'] > 0))  # relu' via post-dropout sign proxy
                # proper relu derivative uses pre-dropout activation sign; recompute cleanly:
                relu_mask = (cache[f'Zn{lp}'] * self.bn[f'gamma{lp}'] + self.bn[f'beta{lp}']) > 0
                dZbn = dA_prev * relu_mask
                grads[f'gamma{lp}'] = (dZbn * cache[f'Zn{lp}']).sum(axis=0, keepdims=True)
                grads[f'beta{lp}']  = dZbn.sum(axis=0, keepdims=True)
                std_inv = 1.0 / np.sqrt(cache[f'var{lp}'] + 1e-8)
                dZn = dZbn * self.bn[f'gamma{lp}']
                N = m
                dZ_prev = (1.0 / N) * std_inv * (
                    N * dZn - dZn.sum(axis=0, keepdims=True)
                    - cache[f'Zn{lp}'] * (dZn * cache[f'Zn{lp}']).sum(axis=0, keepdims=True)
                )
                dZ = dZ_prev
        return grads

    def step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        all_params = {**self.params, **self.bn}
        for k in grads:
            if k not in self.m:
                continue
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
            if k in self.params:
                self.params[k] -= update
            else:
                self.bn[k] -= update

    def fit(self, X, y, X_val=None, y_val=None, epochs=150, batch_size=64,
             patience=15, verbose=False):
        n = X.shape[0]
        best_val_auc = -1
        best_state = None
        no_improve = 0
        history = []
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            Xs, ys = X[perm], y[perm]
            for i in range(0, n, batch_size):
                xb = Xs[i:i + batch_size]
                yb = ys[i:i + batch_size]
                y_hat, cache = self.forward(xb, training=True)
                grads = self.backward(cache, yb)
                self.step(grads)
            train_pred, _ = self.forward(X, training=False)
            train_loss = self.compute_loss(train_pred, y)
            if X_val is not None:
                val_pred, _ = self.forward(X_val, training=False)
                val_auc = roc_auc_score(y_val, val_pred.ravel())
                history.append((epoch, train_loss, val_auc))
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.copy() for k, v in {**self.params, **self.bn}.items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if verbose and epoch % 20 == 0:
                    print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_auc={val_auc:.4f}")
                if no_improve >= patience:
                    break
        if best_state is not None:
            for k, v in best_state.items():
                if k in self.params:
                    self.params[k] = v
                else:
                    self.bn[k] = v
        return history

    def predict_proba(self, X):
        p, _ = self.forward(X, training=False)
        return p.ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

