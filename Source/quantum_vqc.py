"""
quantum_vqc.py
A from-scratch statevector simulator for a small variational quantum circuit
(no qiskit/pennylane available in this environment / no internet access to
install them), plus a Variational Quantum Classifier (VQC) trained with the
Adam optimizer using the parameter-shift rule for exact gradients.

Circuit design (n_qubits, e.g. 4):
  1. Angle encoding:  RY(x_i) on qubit i  for each of the n_qubits input features
  2. Variational layers (repeated `depth` times):
       - RY(theta) single-qubit rotations on every qubit
       - a ring of CNOT entangling gates (0->1->2->...->n-1->0)
  3. Readout: expectation value <Z> on qubit 0 -> mapped to a probability via
     a sigmoid, i.e.  p(essential) = sigmoid(scale * <Z_0>)

Gradients of every rotation angle (encoding is fixed/data, variational
angles are trainable) are computed with the parameter-shift rule:
    d<Z>/dtheta = ( <Z>(theta + pi/2) - <Z>(theta - pi/2) ) / 2
which is EXACT for Pauli rotation gates, then combined with the chain rule
through the sigmoid + binary cross-entropy loss, and used to update the
mean parameters via the Adam optimizer.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------
# Statevector simulator primitives
# ----------------------------------------------------------------------
def ry_matrix(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    state = state.reshape([2] * n_qubits)
    state = np.moveaxis(state, qubit, 0)
    state = np.tensordot(gate, state, axes=([1], [0]))
    state = np.moveaxis(state, 0, qubit)
    return state.reshape(-1)


def apply_cnot(state, control, target, n_qubits):
    state = state.reshape([2] * n_qubits)
    state = np.moveaxis(state, [control, target], [0, 1])
    out = state.copy()
    out[1] = state[1, ::-1]           # flip target when control == 1
    out = np.moveaxis(out, [0, 1], [control, target])
    return out.reshape(-1)


def expectation_z(state, qubit, n_qubits):
    probs = np.abs(state) ** 2
    probs = probs.reshape([2] * n_qubits)
    p0 = probs.take(0, axis=qubit).sum()
    p1 = probs.take(1, axis=qubit).sum()
    return p0 - p1


# ----------------------------------------------------------------------
# Variational Quantum Classifier
# ----------------------------------------------------------------------
class VQC:
    def __init__(self, n_qubits=4, depth=2, scale=2.0, lr=0.05, seed=42):
        self.n_qubits = n_qubits
        self.depth = depth
        self.scale = scale
        self.lr = lr
        rng = np.random.RandomState(seed)
        # one trainable RY angle per qubit per layer
        self.theta = rng.uniform(-0.1, 0.1, size=(depth, n_qubits))
        # Adam state
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0

    def _circuit(self, x, theta):
        """Run the encoding + variational circuit for one sample; return <Z_0>."""
        n = self.n_qubits
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0
        # angle encoding
        for q in range(n):
            state = apply_single_qubit_gate(state, ry_matrix(x[q]), q, n)
        # variational layers
        for l in range(self.depth):
            for q in range(n):
                state = apply_single_qubit_gate(state, ry_matrix(theta[l, q]), q, n)
            for q in range(n - 1):
                state = apply_cnot(state, q, q + 1, n)
            state = apply_cnot(state, n - 1, 0, n)  # close the entangling ring
        return expectation_z(state, 0, n)

    def _predict_expZ(self, X, theta=None):
        theta = self.theta if theta is None else theta
        return np.array([self._circuit(x, theta) for x in X])

    def predict_proba(self, X):
        expz = self._predict_expZ(X)
        return 1.0 / (1.0 + np.exp(-self.scale * expz))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def _param_shift_grad(self, X, y, class_weight):
        """Exact parameter-shift gradient of the BCE loss w.r.t. every theta,
        combined with an Adam update."""
        shift = np.pi / 2
        grad = np.zeros_like(self.theta)
        base_expz = self._predict_expZ(X)
        proba = 1.0 / (1.0 + np.exp(-self.scale * base_expz))
        w = np.where(y == 1, class_weight, 1.0)
        # dLoss/dExpZ  (chain rule through sigmoid + weighted BCE)
        dL_dp = w * (proba - y) / len(y)          # d(BCE)/d(sigmoid input) simplifies to (p - y)
        dL_dexpz = dL_dp * self.scale              # sigmoid'(z)*scale folded via standard BCE+sigmoid identity

        for l in range(self.depth):
            for q in range(self.n_qubits):
                theta_plus = self.theta.copy(); theta_plus[l, q] += shift
                theta_minus = self.theta.copy(); theta_minus[l, q] -= shift
                expz_plus = self._predict_expZ(X, theta_plus)
                expz_minus = self._predict_expZ(X, theta_minus)
                dexpz_dtheta = (expz_plus - expz_minus) / 2.0
                grad[l, q] = np.sum(dL_dexpz * dexpz_dtheta)
        return grad

    def fit(self, X, y, X_val=None, y_val=None, epochs=25, batch_size=32,
            class_weight=1.0, patience=6, verbose=False):
        n = X.shape[0]
        best_val_auc, best_theta, no_improve = -1, self.theta.copy(), 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            Xs, ys = X[perm], y[perm]
            for i in range(0, n, batch_size):
                xb, yb = Xs[i:i + batch_size], ys[i:i + batch_size]
                grad = self._param_shift_grad(xb, yb, class_weight)
                self.t += 1
                self.m = beta1 * self.m + (1 - beta1) * grad
                self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
                m_hat = self.m / (1 - beta1 ** self.t)
                v_hat = self.v / (1 - beta2 ** self.t)
                self.theta -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            if X_val is not None:
                val_auc = roc_auc_score(y_val, self.predict_proba(X_val))
                if val_auc > best_val_auc:
                    best_val_auc, best_theta, no_improve = val_auc, self.theta.copy(), 0
                else:
                    no_improve += 1
                if verbose:
                    print(f"  [VQC] epoch {epoch:2d}  val_auc={val_auc:.4f}")
                if no_improve >= patience:
                    break
        self.theta = best_theta
        return best_val_auc
