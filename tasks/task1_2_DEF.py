"""
Task 1.2 - Centralized Classification via Logistic Regression
=============================================================

Feature mappings:
  Parabola  : ϕ(x) = [x1, x2, x1²]
  Hyperbola : ϕ(x) = [x1, x2, x1·x2]
"""

import numpy as np
import parameters as par


# ─────────────────────────────────────────────
# 1.  Separating Function Rule
# ─────────────────────────────────────────────

def phi_parabola(X):
    """ϕ(x) = [x1, x2, x1²]"""
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0] ** 2])
 
 
def phi_hyperbola(X):
    """ϕ(x) = [x1, x2, x1·x2]"""
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0] * X[:, 1]])

# ─────────────────────────────────────────────
# 2.  Dataset generation
# ─────────────────────────────────────────────

def generate_dataset(M: int, w_true: np.ndarray, b_true: float,
                     phi_fn, noise: float = 0.05, seed: int = 42):
    """Generate M 2-D points labelled by the true separating function."""
    rng    = np.random.default_rng(seed)
    X      = rng.uniform(-3, 3, (M, 2))
    Phi    = phi_fn(X)
    scores = Phi @ w_true + b_true
    labels = np.where(scores >= 0, 1, -1)
    return X, labels

# ─────────────────────────────────────────────
# 3.  Logistic regression loss & gradient
# ─────────────────────────────────────────────

def logistic_loss(wb, Phi, labels):
    """
    L(w,b) = sum_m log(1 + exp(-p_m (w^T ϕ(D_m) + b)))
    wb      : (q+1,) vector  [w; b]
    Phi     : (M, q) mapped features
    labels  : (M,) in {-1, +1}
    """
    w, b = wb[:-1], wb[-1]
    margins = labels * (Phi @ w + b)
    # numerically stable: log(1 + exp(-m)) = log1p(exp(-m))
    return np.sum(np.log1p(np.exp(-margins)))


def logistic_grad(wb, Phi, labels):
    """Gradient of logistic loss w.r.t. [w; b]."""
    w, b = wb[:-1], wb[-1]
    margins = labels * (Phi @ w + b)
    sigma = 1.0 / (1.0 + np.exp(margins))      # sigmoid(-margin)
    dL_dmargin = -sigma                          # d loss / d margin_m
    # d margin_m / d w = p_m ϕ(D_m),  d margin_m / d b = p_m
    weighted = labels * dL_dmargin               # (M,)
    grad_w = Phi.T @ weighted
    grad_b = weighted.sum()
    return np.concatenate([grad_w, [grad_b]])

# ──────────────────────────────────────────────────────────────
#  4. Centralised gradient descent
# ──────────────────────────────────────────────────────────────

def gradient_descent(Phi, labels, alpha: float = 1e-3,
                     max_iter: int = 2000, tol: float = 1e-8,
                     seed: int = 0):
    q   = Phi.shape[1]
    rng = np.random.default_rng(seed)
    wb  = rng.standard_normal(q + 1) * 0.01
 
    cost_hist, grad_norm_hist = [], []
 
    for k in range(max_iter):
        g = logistic_grad(wb, Phi, labels)
        cost_hist.append(logistic_loss(wb, Phi, labels))
        grad_norm_hist.append(np.linalg.norm(g))
        wb -= alpha * g
        if grad_norm_hist[-1] < tol:
            print(f"  Converged at iteration {k+1}")
            break
 
    return wb, cost_hist, grad_norm_hist
 
# ──────────────────────────────────────────────────────────────
#  5. Evaluation
# ──────────────────────────────────────────────────────────────
 
def misclassification_rate(wb, Phi, labels):
    w, b  = wb[:-1], wb[-1]
    preds = np.sign(Phi @ w + b)
    preds[preds == 0] = 1
    return np.mean(preds != labels) * 100
 
 
# ──────────────────────────────────────────────────────────────
#  6. Main task function
# ──────────────────────────────────────────────────────────────
def task1_2():
    M        = par.TASK_1_2_M
    alpha    = par.TASK_1_2_STEPSIZE
    max_iter = par.TASK_1_2_MAX_ITER
 
    mappings = [
        ("Parabola",  phi_parabola),
        ("Hyperbola", phi_hyperbola),
    ]

    results = []
    for map_name, phi_fn in mappings:
        print(f"\n[Task 1.2] Feature mapping: {map_name} | M={M} | α={alpha}")
 
        rng    = np.random.default_rng(99)
        q      = phi_fn(np.zeros((1, 2))).shape[1]
        w_true = rng.standard_normal(q)
        w_true /= np.linalg.norm(w_true)
        b_true = rng.uniform(-0.5, 0.5)
 
        X, labels = generate_dataset(M, w_true, b_true, phi_fn, seed=42)
        Phi       = phi_fn(X)
 
        wb_opt, cost_h, gnorm_h = gradient_descent(
            Phi, labels, alpha=alpha, max_iter=max_iter)
        miss = misclassification_rate(wb_opt, Phi, labels)
        print(f"  Final loss      = {cost_h[-1]:.4f}")
        print(f"  Final grad norm = {gnorm_h[-1]:.2e}")
        print(f"  Misclassified   = {miss:.2f}%")
 
        results.append({
            "map_name":      map_name,
            "cost_hist":     cost_h,
            "grad_norm_hist": gnorm_h,
            "miss_rate":     miss,
        })
 
    return results