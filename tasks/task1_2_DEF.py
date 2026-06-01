#
# Distributed Autonomous Systems
# Task 1.2 - Centralized Classification
# Ivan Colangelo, Nicholas Gioia, Alexandru Zaporojanu
# Bologna, 09/06/26
#

import numpy as np
import Parameters as par
from plots import plot_task_1_2_datasets, plot_task_1_2_metrics, plot_task_1_2_boundary_comparison


# ─────────────────────────────────────────────
# 1.  Separating Function Rule
# ─────────────────────────────────────────────

def phi_parabola(X):
    """ϕ(x) = [x1, x2, x1²]"""
    # X[..., 0] takes first element (the x1 coordinate)  
    x1 = X[..., 0]
    x2 = X[..., 1]
    
    # np.stack with axis=-1 recomposes the data maintaining the correct dimensions
    return np.stack([x1, x2, x1 ** 2], axis=-1)
 
 
def phi_hyperbola(X):
    """ϕ(x) = [x1, x2, x1·x2]"""
    x1 = X[..., 0]
    x2 = X[..., 1]
    return np.stack([x1, x2, x1 * x2], axis=-1)

# ─────────────────────────────────────────────
# 2.  Dataset generation
# ─────────────────────────────────────────────
def generate_dataset(X, w, b, phi_fn):
    """
    Generates binary labels for a dataset using a linear classifier
    in the transformed feature space.
    Given a feature mapping phi_fn, the decision rule is:
        label = +1  if  phi(x) @ w + b >= 0
        label = -1  otherwise

    X      : ndarray of shape (M, d)
             Input data points in the original space.
    w      : array-like of shape (q,)
             Weight vector in the transformed feature space.
    b      : float
             Bias (intercept) term.
    phi_fn : callable
             Feature mapping R^d -> R^q (e.g. phi_parabola, phi_hyperbola).
    
    labels : ndarray of shape (M,) with values in {+1, -1}
             Binary class labels for each input point.
    """
    w = np.array(w)

    Phi = phi_fn(X)
    scores = Phi @ w + b

    labels = np.where(scores >= 0, 1, -1)

    return labels

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
    # d margin_m / d w = p_m ϕ(D_m),  d margin_m / d b = p_m
    weighted = - labels * sigma               # (M,)
    grad_w = Phi.T @ weighted
    grad_b = weighted.sum()
    return np.concatenate([grad_w, [grad_b]])

# ──────────────────────────────────────────────────────────────
#  4. Centralised gradient descent
# ──────────────────────────────────────────────────────────────

def centralized_gradient_descent(X, Phi, labels, stepsize, max_iter, seed=0, tol=1e-8):
    '''Performs centralized gradient descent to minimize logistic loss.
    X        : (M, d) input data points
    Phi      : (M, q) mapped features
    labels   : (M,) binary class labels
    stepsize : float, step size for gradient descent
    max_iter : int, maximum number of iterations
    seed     : int, random seed for initialization
    tol      : float, tolerance for convergence
    Returns:
    wb       : (q+1,) learned parameters [w; b]
    cost_hist : list of logistic loss values at each iteration
    grad_norm_hist : list of gradient norms at each iteration
    '''
    Phi = np.array([Phi(x) for x in X]) # (M, q)

    q_dim   = Phi.shape[1]
    rng = np.random.default_rng(seed)
    wb = rng.standard_normal(q_dim + 1) * 0.01 # small random init for [w; b]
 
    cost_hist, grad_norm_hist = [], []
 
    for k in range(max_iter):
        cost = logistic_loss(wb, Phi, labels)
        grad = logistic_grad(wb, Phi, labels)

        cost_hist.append(cost)
        grad_norm_hist.append(np.linalg.norm(grad))

        wb -= stepsize * grad

        if grad_norm_hist[-1] < tol:
            print(f"  Converged at iteration {k+1}")
            break
 
    return wb, cost_hist, grad_norm_hist
 
# ──────────────────────────────────────────────────────────────
#  6. Main task function
# ──────────────────────────────────────────────────────────────
def task1_2():
    print("\n" + "="*50)
    print("--- Task 1.2: Centralized Classification ---")
    print("="*50 + "\n")

    M        = par.TASK_1_2_M
    range = par.TASK_1_2_RANGE
    stepsize    = par.TASK_1_2_STEPSIZE
    max_iter = par.TASK_1_2_MAX_ITER
    np.random.seed(0)
 
    # Generate random points
    lower, upper = range
    X = np.random.uniform(lower, upper, (M, 2)) # M random points in 2D
    
    # Generate dataset
    print(f" --- Generating dataset with M={M} points in range {range} --- ")
    # Parabola mapping
    labels_parabola = generate_dataset(X, w=par.W_PARABOLA, b=par.B_PARABOLA, phi_fn=phi_parabola)
    print(f"  Parabola dataset: {np.sum(labels_parabola == 1)} positive, {np.sum(labels_parabola == -1)} negative")
    # Hyperbola mapping
    labels_hyperbola = generate_dataset(X, w=par.W_HYPERBOLA, b=par.B_HYPERBOLA, phi_fn=phi_hyperbola)
    print(f"  Hyperbola dataset: {np.sum(labels_hyperbola == 1)} positive, {np.sum(labels_hyperbola == -1)} negative")
    
    # Plot comparison of datasets
    if par.TASK_1_2_FUTURE_MAPPING:
        plot_task_1_2_datasets(X, labels_parabola, labels_hyperbola, phi_parabola, phi_hyperbola, par.W_PARABOLA, par.B_PARABOLA, par.W_HYPERBOLA, par.B_HYPERBOLA, title_prefix="[Initial Datasets]")
    
    # Centralized gradient descent
    print("\n --- Running centralized gradient descent --- ")

    print("\n  - Logistic Regression: Parabola")
    theta_parabola, cost_hist_parabola, grad_norm_hist_parabola = centralized_gradient_descent(X, phi_parabola, labels_parabola, stepsize, max_iter)
    if par.TASK_1_2_METRICS:
        plot_task_1_2_metrics(cost_hist_parabola, grad_norm_hist_parabola, "Parabola Mapping")
    print(f"  Final weights: {theta_parabola[:-1]}, bias: {theta_parabola[-1]:.4f}")

    print("\n  - Logistic Regression: Hyperbola")
    theta_hyperbola, cost_hist_hyperbola, grad_norm_hist_hyperbola = centralized_gradient_descent(X, phi_hyperbola, labels_hyperbola, stepsize, max_iter)
    if par.TASK_1_2_METRICS:
        plot_task_1_2_metrics(cost_hist_hyperbola, grad_norm_hist_hyperbola, "Hyperbola Mapping")
    print(f"  Final weights: {theta_hyperbola[:-1]}, bias: {theta_hyperbola[-1]:.4f}")

    print("\n --- Performance Evaluation --- ")
    # Extract learned weights and biases for both mappings
    w_learned_parabola, b_learned_parabola = theta_parabola[:-1], theta_parabola[-1]
    w_learned_hyperbola, b_learned_hyperbola = theta_hyperbola[:-1], theta_hyperbola[-1]

    # Generate predicted labels using the learned parameters
    pred_labels_parabola = generate_dataset(X, w_learned_parabola, b_learned_parabola, phi_parabola)
    pred_labels_hyperbola = generate_dataset(X, w_learned_hyperbola, b_learned_hyperbola, phi_hyperbola)

    # Calculate and print the accuracy for both mappings
    acc_rate_parabola = np.mean(pred_labels_parabola == labels_parabola) * 100
    acc_rate_hyperbola = np.mean(pred_labels_hyperbola == labels_hyperbola) * 100

    print("\nParabola Mapping:")
    print(f"    - Predicted labels: {np.sum(pred_labels_parabola == 1)} positive, {np.sum(pred_labels_parabola == -1)} negative")
    print(f"    - Missclassified for label 1: {np.sum((pred_labels_parabola == -1) & (labels_parabola == 1))}")
    print(f"    - Missclassified for label -1: {np.sum((pred_labels_parabola == 1) & (labels_parabola == -1))}")
    print(f"    - Accuracy: {acc_rate_parabola:.2f}%")

    print("\nHyperbola Mapping:")
    print(f"    - Predicted labels: {np.sum(pred_labels_hyperbola == 1)} positive, {np.sum(pred_labels_hyperbola == -1)} negative")
    print(f"    - Missclassified for label 1: {np.sum((pred_labels_hyperbola == -1) & (labels_hyperbola == 1))}")
    print(f"    - Missclassified for label -1: {np.sum((pred_labels_hyperbola == 1) & (labels_hyperbola == -1))}")
    print(f"    - Accuracy: {acc_rate_hyperbola:.2f}%")

    if par.TASK_1_2_FUTURE_MAPPING:    
        plot_task_1_2_datasets(X, pred_labels_parabola, pred_labels_hyperbola, phi_parabola, phi_hyperbola, w_learned_parabola, b_learned_parabola, w_learned_hyperbola, b_learned_hyperbola,"Learned Decision Boundaries")
    if par.TASK_1_2_FUTURE_MAPPING_COMPARISON:
        plot_task_1_2_boundary_comparison(X, labels_parabola, labels_hyperbola, phi_parabola, phi_hyperbola, par.W_PARABOLA, par.B_PARABOLA, w_learned_parabola, b_learned_parabola, par.W_HYPERBOLA, par.B_HYPERBOLA, w_learned_hyperbola, b_learned_hyperbola, title_prefix="Boundary Comparison - True vs Learned")