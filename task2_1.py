"""
task2_1.py  –  Aggregative Tracking Algorithm for Multi-Robot Surveillance
===========================================================================

Problem (Task 2.1):
  min_{z ∈ R^{2N}}  Σᵢ ℓᵢ(zᵢ, σ(z))

  with  σ(z) = (1/N) Σᵢ φᵢ(zᵢ),   φᵢ(zᵢ) = zᵢ   (identity)

  ℓᵢ(zᵢ, σ) = γᵢ ‖zᵢ − rᵢ‖²  +  ‖σ − r₀‖²
               ↑ stay near private target    ↑ barycenter near common target

Gradients (used in the algorithm):
  ∇₁ℓᵢ(zᵢ, σ) = 2γᵢ (zᵢ − rᵢ)          w.r.t. zᵢ
  ∇₂ℓᵢ(zᵢ, σ) = 2 (σ − r₀)              w.r.t. σ
  ∇φᵢ(zᵢ)     = I₂                       Jacobian of identity map

Algorithm (Aggregative Tracking):
  zᵢᵏ⁺¹ = zᵢᵏ − α (∇₁ℓᵢ(zᵢᵏ, sᵢᵏ) + ∇φᵢ(zᵢᵏ) vᵢᵏ)
  sᵢᵏ⁺¹ = Σⱼ aᵢⱼ sⱼᵏ + φᵢ(zᵢᵏ⁺¹) − φᵢ(zᵢᵏ)          s⁰ᵢ = φᵢ(z⁰ᵢ)
  vᵢᵏ⁺¹ = Σⱼ aᵢⱼ vⱼᵏ + ∇₂ℓᵢ(zᵢᵏ⁺¹,sᵢᵏ⁺¹) − ∇₂ℓᵢ(zᵢᵏ,sᵢᵏ)  v⁰ᵢ = ∇₂ℓᵢ(z⁰ᵢ,s⁰ᵢ)
"""

import numpy as np
import Parameters as par
from tasks.task1_1_DEF import build_metropolis_weights
import networkx as nx


# ──────────────────────────────────────────────────────────────
#  Cost, gradients, and φ for the surveillance problem
# ──────────────────────────────────────────────────────────────

def phi_i(zi):
    """φᵢ(zᵢ) = zᵢ  (identity map, returns R²)."""
    return zi.copy()


def grad_phi_i(zi):
    """∇φᵢ(zᵢ) = I₂  (2x2 identity Jacobian)."""
    return np.eye(len(zi))


def local_cost(zi, sigma, r_i, gamma_i, r0):
    """ℓᵢ(zᵢ, σ) = γᵢ‖zᵢ-rᵢ‖² + ‖σ−r₀‖²"""
    return gamma_i * np.dot(zi - r_i, zi - r_i) + np.dot(sigma - r0, sigma - r0)


def grad1_li(zi, sigma, r_i, gamma_i):
    """∇₁ℓᵢ  w.r.t. zᵢ  =  2γᵢ(zᵢ − rᵢ)"""
    return 2.0 * gamma_i * (zi - r_i)


def grad2_li(zi, sigma, r0):
    """∇₂ℓᵢ  w.r.t. σ   =  2(σ − r₀)"""
    return 2.0 * (sigma - r0)


# ──────────────────────────────────────────────────────────────
#  Optimal solution (closed-form for quadratic cost)
# ──────────────────────────────────────────────────────────────

def compute_optimal(r_targets, gammas, r0):
    """
    Closed-form optimum for  Σᵢ [γᵢ‖zᵢ−rᵢ‖² + ‖σ−r₀‖²].

    Setting gradient to zero:
      2γᵢ(zᵢ* − rᵢ) + (2/N)(σ* − r₀) = 0  for all i
    → zᵢ* = rᵢ − (1/(Nγᵢ))(σ* − r₀)

    Summing over i and dividing by N:
      σ* = (1/N)Σzᵢ* = r̄ − (Σ 1/(Nγᵢ))/N · (σ* − r₀)

    Let  C = (1/N²) Σ(1/γᵢ),  r̄ = (1/N)Σrᵢ
      σ*(1+C) = r̄ + C·r₀  →  σ* = (r̄ + C·r₀)/(1+C)
    """
    N    = len(gammas)
    r_bar = np.mean(r_targets, axis=0)
    C    = np.sum(1.0 / gammas) / N**2
    sigma_opt = (r_bar + C * r0) / (1.0 + C)
    z_opt = r_targets - (sigma_opt - r0)[None, :] / (N * gammas[:, None])
    return z_opt, sigma_opt


# ──────────────────────────────────────────────────────────────
#  Core algorithm
# ──────────────────────────────────────────────────────────────

def run_aggregative_tracking(z_init, r_targets, gammas, r0, W,
                              alpha=0.01, max_iter=1000):
    """
    Run the Aggregative Tracking algorithm.

    Parameters
    ----------
    z_init   : (N, 2) initial positions of robots
    r_targets: (N, 2) private target of each robot
    gammas   : (N,)   trade-off weights γᵢ > 0
    r0       : (2,)   common target to protect (barycenter target)
    W        : (N, N) Metropolis weight matrix
    alpha    : float  step size
    max_iter : int    number of iterations

    Returns
    -------
    z_hist   : (max_iter+1, N, 2) trajectory of all robots
    s_hist   : (max_iter+1, N, 2) trajectory of s estimates
    metrics  : dict with 'cost', 'grad_norm', 'consensus', 'sigma_error'
    """
    N = z_init.shape[0]

    # ── Initialisation ──────────────────────────────────────
    z = z_init.copy()          # (N, 2)
    s = z.copy()               # s⁰ᵢ = φᵢ(z⁰ᵢ) = zᵢ⁰
    v = np.zeros((N, 2))
    for i in range(N):
        v[i] = grad2_li(z[i], s[i], r0)   # v⁰ᵢ = ∇₂ℓᵢ(z⁰ᵢ, s⁰ᵢ)

    z_hist = np.zeros((max_iter + 1, N, 2))
    s_hist = np.zeros((max_iter + 1, N, 2))
    z_hist[0] = z
    s_hist[0] = s

    cost_hist       = []
    grad_norm_hist  = []
    consensus_hist  = []
    sigma_error_hist = []

    for k in range(max_iter):

        # ── z update ─────────────────────────────────────────
        z_new = np.zeros_like(z)
        for i in range(N):
            g1 = grad1_li(z[i], s[i], r_targets[i], gammas[i])
            Jphi = grad_phi_i(z[i])          # I₂
            z_new[i] = z[i] - alpha * (g1 + Jphi @ v[i])

        # ── s update (consensus + φ tracking) ────────────────
        s_new = W @ s  +  np.array([phi_i(z_new[i]) - phi_i(z[i])
                                     for i in range(N)])

        # ── v update (gradient-of-σ tracking) ────────────────
        v_new = W @ v  +  np.array([
            grad2_li(z_new[i], s_new[i], r0) - grad2_li(z[i], s[i], r0)
            for i in range(N)])

        # ── Metrics ──────────────────────────────────────────
        sigma_true = np.mean(z, axis=0)
        total_cost = sum(local_cost(z[i], sigma_true, r_targets[i], gammas[i], r0)
                         for i in range(N))
        cost_hist.append(total_cost)

        # Full gradient norm: ∇zᵢ Σⱼ ℓⱼ = ∇₁ℓᵢ + (1/N) ∇φᵢᵀ Σⱼ ∇₂ℓⱼ
        # Since φᵢ = I, ∇φᵢ = I, this simplifies to:
        #   ∇₁ℓᵢ(zᵢ,σ) + (1/N) Σⱼ ∇₂ℓⱼ(zⱼ,σ)
        sum_grad2 = sum(grad2_li(z[i], sigma_true, r0) for i in range(N))
        total_grad = np.array([
            grad1_li(z[i], sigma_true, r_targets[i], gammas[i]) + sum_grad2 / N
            for i in range(N)])
        grad_norm_hist.append(np.linalg.norm(total_grad))

        # Consensus: deviation of s estimates from true barycenter
        sigma_hat = np.mean(s, axis=0)   # average of s estimates
        consensus_hist.append(np.linalg.norm(s - sigma_hat))

        # σ estimation error: how close is each sᵢ to true barycenter
        sigma_error_hist.append(np.linalg.norm(s - sigma_true))

        # ── Advance ──────────────────────────────────────────
        z, s, v = z_new, s_new, v_new
        z_hist[k + 1] = z
        s_hist[k + 1] = s

    metrics = {
        "cost":        cost_hist,
        "grad_norm":   grad_norm_hist,
        "consensus":   consensus_hist,
        "sigma_error": sigma_error_hist,
    }
    return z_hist, s_hist, metrics


# ──────────────────────────────────────────────────────────────
#  Scenario builder helpers
# ──────────────────────────────────────────────────────────────

def _build_graph(graph_type, N):
    gt = graph_type.lower()
    if gt == "cycle":
        return nx.cycle_graph(N)
    elif gt == "path":
        return nx.path_graph(N)
    elif gt == "star":
        return nx.star_graph(N - 1)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def make_scenario(N, graph_type="cycle", seed=0, spread=5.0):
    """
    Build a random scenario: initial positions, private targets, common target.

    Returns
    -------
    z_init, r_targets, gammas, r0, W, G
    """
    rng = np.random.default_rng(seed)

    # Robots start on a rough circle with some noise
    angles  = 2 * np.pi * np.arange(N) / N
    z_init  = spread * np.column_stack([np.cos(angles), np.sin(angles)])
    z_init += rng.uniform(-0.5, 0.5, z_init.shape)

    # Private targets scattered around the arena
    r_targets = rng.uniform(-spread * 1.5, spread * 1.5, (N, 2))

    # Common target near origin
    r0 = rng.uniform(-1.0, 1.0, 2)

    # Trade-off weights
    gammas = rng.uniform(0.5, 2.0, N)

    G = _build_graph(graph_type, N)
    W = build_metropolis_weights(G)

    return z_init, r_targets, gammas, r0, W, G


# ──────────────────────────────────────────────────────────────
#  Main task function
# ──────────────────────────────────────────────────────────────

def run_task2_1():
    """
    Run Task 2.1 – Aggregative Tracking for multi-robot surveillance.

    Runs three sub-scenarios:
      A) baseline (cycle graph, default γ)
      B) different γ values (more cohesion vs. more target-chasing)
      C) different target locations

    Returns
    -------
    scenarios : list of dicts, each with keys:
        'label', 'N', 'z_hist', 's_hist', 'z_init', 'r_targets',
        'gammas', 'r0', 'G', 'W', 'z_opt', 'sigma_opt', 'metrics'
    """
    N        = par.TASK_2_1_N
    alpha    = par.TASK_2_1_ALPHA
    max_iter = par.TASK_2_1_MAX_ITER
    graph_type = par.TASK_2_1_GRAPH

    print(f"\n[Task 2.1] N={N} | α={alpha} | max_iter={max_iter} | graph={graph_type}")

    scenarios = []

    # ── Scenario A: baseline ────────────────────────────────
    print("\n  Scenario A – Baseline (balanced γ, cycle graph)")
    z_init, r_targets, gammas, r0, W, G = make_scenario(N, graph_type, seed=0)
    z_hist, s_hist, metrics = run_aggregative_tracking(
        z_init, r_targets, gammas, r0, W, alpha=alpha, max_iter=max_iter)
    z_opt, sigma_opt = compute_optimal(r_targets, gammas, r0)

    print(f"    Final cost:      {metrics['cost'][-1]:.4f}")
    print(f"    Final |grad|:    {metrics['grad_norm'][-1]:.2e}")
    print(f"    σ estimate err:  {metrics['sigma_error'][-1]:.2e}")
    print(f"    ‖z_final - z*‖:  {np.linalg.norm(z_hist[-1] - z_opt):.2e}")

    scenarios.append(dict(label="Baseline (balanced γ)",
                          N=N, z_hist=z_hist, s_hist=s_hist,
                          z_init=z_init, r_targets=r_targets, gammas=gammas,
                          r0=r0, G=G, W=W, z_opt=z_opt,
                          sigma_opt=sigma_opt, metrics=metrics))

    # ── Scenario B: high γ (formation tight, less target-chasing) ──
    print("\n  Scenario B – High γ (tight formation)")
    z_init_b, r_targets_b, _, r0_b, W_b, G_b = make_scenario(N, graph_type, seed=1)
    gammas_b = np.full(N, 5.0)   # high γ → formation dominates
    z_hist_b, s_hist_b, metrics_b = run_aggregative_tracking(
        z_init_b, r_targets_b, gammas_b, r0_b, W_b, alpha=alpha, max_iter=max_iter)
    z_opt_b, sigma_opt_b = compute_optimal(r_targets_b, gammas_b, r0_b)

    print(f"    Final cost:      {metrics_b['cost'][-1]:.4f}")
    print(f"    Final |grad|:    {metrics_b['grad_norm'][-1]:.2e}")

    scenarios.append(dict(label="High γ (tight formation)",
                          N=N, z_hist=z_hist_b, s_hist=s_hist_b,
                          z_init=z_init_b, r_targets=r_targets_b, gammas=gammas_b,
                          r0=r0_b, G=G_b, W=W_b, z_opt=z_opt_b,
                          sigma_opt=sigma_opt_b, metrics=metrics_b))

    # ── Scenario C: low γ (target-chasing dominates) ────────
    print("\n  Scenario C – Low γ (target-chasing)")
    z_init_c, r_targets_c, _, r0_c, W_c, G_c = make_scenario(N, graph_type, seed=2)
    gammas_c = np.full(N, 0.1)   # low γ → barycenter-to-r₀ dominates
    z_hist_c, s_hist_c, metrics_c = run_aggregative_tracking(
        z_init_c, r_targets_c, gammas_c, r0_c, W_c, alpha=alpha, max_iter=max_iter)
    z_opt_c, sigma_opt_c = compute_optimal(r_targets_c, gammas_c, r0_c)

    print(f"    Final cost:      {metrics_c['cost'][-1]:.4f}")
    print(f"    Final |grad|:    {metrics_c['grad_norm'][-1]:.2e}")

    scenarios.append(dict(label="Low γ (target-chasing)",
                          N=N, z_hist=z_hist_c, s_hist=s_hist_c,
                          z_init=z_init_c, r_targets=r_targets_c, gammas=gammas_c,
                          r0=r0_c, G=G_c, W=W_c, z_opt=z_opt_c,
                          sigma_opt=sigma_opt_c, metrics=metrics_c))

    return scenarios
