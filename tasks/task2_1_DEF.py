import numpy as np
import Parameters as par
from graph_utils import get_graph_and_matrix


# =============================================================================
# 1.  LOCAL COST AND GRADIENTS
# =============================================================================

def local_cost(zi, sigma, r_i, b_i, gamma_i, beta_i):
    """ℓᵢ(zᵢ, σ) = γᵢ‖zᵢ − rᵢ‖² + βᵢ‖zᵢ − σ − bᵢ‖²"""
    return (gamma_i * np.dot(zi - r_i, zi - r_i)
            + beta_i * np.dot(zi - sigma - b_i, zi - sigma - b_i))


def grad1_li(zi, sigma, r_i, b_i, gamma_i, beta_i):
    """∇₁ℓᵢ  w.r.t. zᵢ  =  2γᵢ(zᵢ − rᵢ) + 2βᵢ(zᵢ − σ − bᵢ)"""
    return (2.0 * gamma_i * (zi - r_i)
            + 2.0 * beta_i * (zi - sigma - b_i))


def grad2_li(zi, sigma, b_i, beta_i):
    """∇₂ℓᵢ  w.r.t. σ  =  −2βᵢ(zᵢ − σ − bᵢ)"""
    return -2.0 * beta_i * (zi - sigma - b_i)


def phi(z_i):
    """Aggregation map  φ(zᵢ) = zᵢ  (identity)."""
    return z_i


def grad_phi(z_i):
    """Jacobian of φ: scalar 1.0 (identity map)."""
    return 1.0


# =============================================================================
# 2.  METRICS COMPUTATION (post-run)
# =============================================================================

def _compute_metrics(z, r, b, gamma, beta, N, maxK):
    """
    Compute per-iteration metrics from the full trajectory history z.

    Returns
    -------
    cost      : (maxK,) total cost J(z, σ)
    grad_norm : (maxK,) norm of the global gradient ∇J
    consensus : (maxK,) consensus error ‖z − z̄‖
    sigma_err : (maxK,) σ estimation error ‖σ(z) − σ̄‖ where σ̄ = mean at last iter
    """
    cost      = np.zeros(maxK)
    grad_norm = np.zeros(maxK)
    consensus = np.zeros(maxK)
    sigma_err = np.zeros(maxK)

    sigma_opt = np.mean(z[-1], axis=0)  # reference: barycenter at convergence

    for k in range(maxK):
        sigma_k = np.mean(z[k], axis=0)

        cost_k     = 0.0
        grad_sq    = 0.0
        sum_grad2  = np.sum(
            [-2.0 * beta[j] * (z[k, j] - sigma_k - b[j]) for j in range(N)],
            axis=0,
        )

        for i in range(N):
            cost_k += local_cost(z[k, i], sigma_k, r[i], b[i], gamma[i], beta[i])

            # ∇_{z_i} J = ∇₁ℓᵢ + (1/N) Σⱼ ∇₂ℓⱼ
            grad_i = (
                2.0 * gamma[i] * (z[k, i] - r[i])
                + 2.0 * beta[i] * (z[k, i] - sigma_k - b[i])
                + (1.0 / N) * sum_grad2
            )
            grad_sq += np.dot(grad_i, grad_i)

        cost[k]      = cost_k
        grad_norm[k] = np.sqrt(grad_sq)
        consensus[k] = np.linalg.norm(z[k] - sigma_k)          # ‖z − z̄‖
        sigma_err[k] = np.linalg.norm(sigma_k - sigma_opt)      # ‖σₖ − σ*‖

    return cost, grad_norm, consensus, sigma_err


# =============================================================================
# 3.  AGGREGATIVE TRACKING ALGORITHM
# =============================================================================

def _run_scenario(graph_type, N, stepsize, maxK, gamma, beta, b, r, z_init, label):
    """
    Run one aggregative-tracking scenario and return a result dict compatible
    with the plotting functions in plots_task2_DEF.py.

    Parameters
    ----------
    graph_type : str   – graph topology ('cycle', 'path', 'star', …)
    N          : int   – number of robots
    stepsize   : float – gradient step α
    maxK       : int   – number of iterations
    gamma      : (N,)  – private-target weights γᵢ
    beta       : (N,)  – formation weights βᵢ
    b          : (N,2) – formation offset vectors bᵢ
    r          : (N,2) – private target positions rᵢ
    z_init     : (N,2) – initial robot positions
    label      : str   – human-readable scenario name (for plot titles)

    Returns
    -------
    dict with keys: label, N, graph_type, r_targets, z_init, z_opt,
                    sigma_opt, z_hist, metrics
    """
    G, A = get_graph_and_matrix(N, graph_type)
    Adj  = (A - np.eye(N)) > 0  # boolean neighbour matrix (off-diagonal)

    # State arrays: z[k,i] position, s[k,i] σ tracker, v[k,i] grad tracker
    z = np.zeros((maxK, N, 2))
    s = np.zeros((maxK, N, 2))
    v = np.zeros((maxK, N, 2))

    z[0] = z_init.copy()
    for i in range(N):
        s[0, i] = phi(z[0, i])
        v[0, i] = grad2_li(z[0, i], s[0, i], b[i], beta[i])

    # ── Main loop ────────────────────────────────────────────────────────
    for k in range(maxK - 1):

        # z-update
        for i in range(N):
            g1    = grad1_li(z[k, i], s[k, i], r[i], b[i], gamma[i], beta[i])
            g_phi = grad_phi(z[k, i])
            z[k+1, i] = z[k, i] - stepsize * (g1 + g_phi * v[k, i])

        # s-update  (σ tracker)
        for i in range(N):
            neighbours = np.where(Adj[i])[0]
            s[k+1, i]  = A[i, i] * s[k, i]
            for j in neighbours:
                s[k+1, i] += A[i, j] * s[k, j]
            s[k+1, i] += phi(z[k+1, i]) - phi(z[k, i])

        # v-update  (gradient tracker)
        for i in range(N):
            neighbours = np.where(Adj[i])[0]
            v[k+1, i]  = A[i, i] * v[k, i]
            for j in neighbours:
                v[k+1, i] += A[i, j] * v[k, j]
            grad2_new   = grad2_li(z[k+1, i], s[k+1, i], b[i], beta[i])
            grad2_old   = grad2_li(z[k,   i], s[k,   i], b[i], beta[i])
            v[k+1, i]  += grad2_new - grad2_old

    # ── Metrics ──────────────────────────────────────────────────────────
    cost, grad_norm, consensus, sigma_err = _compute_metrics(
        z, r, b, gamma, beta, N, maxK
    )

    z_opt     = z[-1]                       # (N, 2)
    sigma_opt = np.mean(z_opt, axis=0)      # (2,)

    print(f"  [{label}]  Final cost: {cost[-1]:.4f} | "
          f"Final |∇J|: {grad_norm[-1]:.2e} | "
          f"Consensus: {consensus[-1]:.2e}")

    return {
        "label":      label,
        "N":          N,
        "graph_type": graph_type,
        "r_targets":  r,           # (N, 2) private targets
        "z_init":     z_init,      # (N, 2) initial positions
        "z_opt":      z_opt,       # (N, 2) final positions
        "sigma_opt":  sigma_opt,   # (2,)   optimal barycenter
        "z_hist":     z,           # (maxK, N, 2) full trajectory
        "metrics": {
            "cost":        cost,
            "grad_norm":   grad_norm,
            "consensus":   consensus,
            "sigma_error": sigma_err,
        },
    }


# =============================================================================
# 4.  MAIN TASK FUNCTION  (called from main.py)
# =============================================================================

def run_task2_1():
    """
    Run Task 2.1 for the scenario(s) defined in Parameters.py.

    Returns
    -------
    scenarios : list of result dicts (one per scenario), each compatible
                with the plotting functions in plots_task2_DEF.py.
    """
    N        = par.TASK_2_1_N
    stepsize = par.TASK_2_1_ALPHA
    maxK     = par.TASK_2_1_MAX_ITER
    graph    = par.TASK_2_1_GRAPH

    # ── Shared problem setup ─────────────────────────────────────────────
    # Weights
    gamma = np.ones(N) * 1.0   # private-target weight γᵢ
    beta  = np.ones(N) * 0.1   # formation weight βᵢ

    # Formation offsets b_i arranged as a regular polygon  (Σ bᵢ = 0)
    radius = 6.0
    b = np.array([
        [radius * np.cos(2.0 * np.pi * i / N),
         radius * np.sin(2.0 * np.pi * i / N)]
        for i in range(N)
    ])

    # Private targets – reproducible random positions
    rng = np.random.default_rng(0)
    r   = 10.0 * (rng.random((N, 2)) - 0.5)

    # Common initial position (all robots start at the same point)
    z_init = np.tile(np.array([-5.0, 5.0]), (N, 1)).astype(float)

    # ── Run scenarios ────────────────────────────────────────────────────
    scenarios = []

    # Baseline: graph topology from Parameters.py
    print(f"\n[Task 2.1] Running baseline scenario – graph={graph}, N={N}, "
          f"α={stepsize}, K={maxK}")
    sc_baseline = _run_scenario(
        graph_type=graph,
        N=N, stepsize=stepsize, maxK=maxK,
        gamma=gamma, beta=beta, b=b, r=r, z_init=z_init,
        label=f"{graph.capitalize()} graph",
    )
    scenarios.append(sc_baseline)

    # Optional: topology comparison (uncomment in Parameters.py to enable)
    # If TASK_2_1_COMPARE_GRAPHS is defined and True, run all three topologies
    compare = getattr(par, "TASK_2_1_COMPARE_GRAPHS", False)
    if compare:
        for g_type in ["cycle", "path", "star"]:
            if g_type == graph:
                continue  # already done above
            print(f"\n[Task 2.1] Comparison scenario – graph={g_type}")
            sc = _run_scenario(
                graph_type=g_type,
                N=N, stepsize=stepsize, maxK=maxK,
                gamma=gamma, beta=beta, b=b, r=r, z_init=z_init,
                label=f"{g_type.capitalize()} graph",
            )
            scenarios.append(sc)

    return scenarios