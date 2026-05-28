"""
Task 2.1: Distributed Aggregative Tracking
"""
import numpy as np
import Parameters as par
from graph_utils import get_graph_and_matrix

def phi(z_i):
    return z_i

def grad_phi(z_i):
    return 1.0

def local_cost(zi, sigma, r_i, gamma_i, beta_i):
    """Calculates the local cost without the 'b' formation parameter."""
    return gamma_i * np.dot(zi - r_i, zi - r_i) + beta_i * np.dot(zi - sigma, zi - sigma)

def grad1_li(zi, sigma, r_i, gamma_i, beta_i):
    """Gradient of the local cost with respect to zi."""
    return 2.0 * gamma_i * (zi - r_i) + 2.0 * beta_i * (zi - sigma)

def grad2_li(zi, sigma, beta_i):
    """Gradient of the local cost with respect to sigma."""
    return -2.0 * beta_i * (zi - sigma)

def _compute_metrics(z, r_targets, gamma, beta, N, max_iters):
    """Computes global metrics for the team performance."""
    cost = np.zeros(max_iters)
    grad_norm = np.zeros(max_iters)
    consensus = np.zeros(max_iters)
    sigma_err = np.zeros(max_iters)

    # Calculate analytical optimal barycenter (sigma_opt)
    num = np.zeros(2)
    den_sub = 0.0
    for i in range(N):
        num += (gamma[i] * r_targets[i]) / (gamma[i] + beta[i])
        den_sub += beta[i] / (gamma[i] + beta[i])
    sigma_opt = (num / N) / (1.0 - den_sub / N)

    for k in range(max_iters):
        sigma_k = np.mean(z[k], axis=0)
        c_val = 0.0
        g_val = 0.0
        
        for i in range(N):
            c_val += local_cost(z[k, i], sigma_k, r_targets[i], gamma[i], beta[i])
            g1 = grad1_li(z[k, i], sigma_k, r_targets[i], gamma[i], beta[i])
            g_val += np.linalg.norm(g1)**2
            
        cost[k] = c_val
        grad_norm[k] = np.sqrt(g_val)
        sigma_err[k] = np.linalg.norm(sigma_k - sigma_opt)
        
    return cost, grad_norm, consensus, sigma_err

def run_task2_1():
    print("======================================================================")
    print("--- Starting Task 2.1: Aggregative Tracking ---")
    print("======================================================================")
    
    N = par.TASK_2_1_N
    max_iters = getattr(par, 'TASK_2_1_MAX_ITER', 15000)
    stepsize = getattr(par, 'TASK_2_1_ALPHA', 0.01)
    gamma_val = getattr(par, 'TASK_2_1_GAMMA', 1.0)
    beta_val = getattr(par, 'TASK_2_1_BETA', 0.1)

    # 1. Start Positions: Circle on the left with random noise
    np.random.seed(42)
    init_center = np.array([-7.0, 0.0])
    init_radius = 1.0
    z_init = np.array([
        [init_center[0] + init_radius * np.cos(2.0 * np.pi * i / N),
         init_center[1] + init_radius * np.sin(2.0 * np.pi * i / N)]
        for i in range(N)
    ]) + np.random.uniform(-0.5, 0.5, (N, 2))

    # 2. Target Positions: Hexagon on the right
    target_center = np.array([7.0, 0.0])
    target_radius = 4.0
    r_targets = np.array([
        [target_center[0] + target_radius * np.cos(2.0 * np.pi * i / N),
         target_center[1] + target_radius * np.sin(2.0 * np.pi * i / N)]
        for i in range(N)
    ])

    gamma = np.ones(N) * gamma_val
    beta = np.ones(N) * beta_val

    # 3. Calculate Analytical Optima
    num = np.zeros(2)
    den_sub = 0.0
    for i in range(N):
        num += (gamma[i] * r_targets[i]) / (gamma[i] + beta[i])
        den_sub += beta[i] / (gamma[i] + beta[i])
    sigma_opt = (num / N) / (1.0 - den_sub / N)

    z_opt = np.zeros((N, 2))
    for i in range(N):
        z_opt[i] = (gamma[i] * r_targets[i] + beta[i] * sigma_opt) / (gamma[i] + beta[i])

    graphs_to_test = getattr(par, 'TASK_2_1_GRAPHS', ["cycle", "path", "star"])
    scenarios = []

    for graph_type in graphs_to_test:
        print(f"\n[Task 2.1] Running scenario – graph={graph_type}, N={N}, α={stepsize}, K={max_iters}")
        G, A = get_graph_and_matrix(N, graph_type)
        Adj = (A - np.eye(N)) > 0
        
        z = np.zeros((max_iters, N, 2))
        s = np.zeros((max_iters, N, 2))
        v = np.zeros((max_iters, N, 2))
        
        z[0] = z_init.copy()
        for i in range(N):
            s[0, i] = phi(z[0, i])
            v[0, i] = grad2_li(z[0, i], s[0, i], beta[i])

        for k in range(max_iters - 1):
            z_next = np.zeros((N, 2))
            # Z-Update
            for i in range(N):
                g1 = grad1_li(z[k, i], s[k, i], r_targets[i], gamma[i], beta[i])
                g_phi = grad_phi(z[k, i])
                u_nom = -stepsize * (g1 + g_phi * v[k, i])
                z_next[i] = z[k, i] + u_nom
                
            # S and V Updates
            for i in range(N):
                neighbours = np.where(Adj[i])[0]
                
                s[k+1, i] = A[i, i] * s[k, i]
                for j in neighbours:
                    s[k+1, i] += A[i, j] * s[k, j]
                s[k+1, i] += phi(z_next[i]) - phi(z[k, i])

                v[k+1, i] = A[i, i] * v[k, i]
                for j in neighbours:
                    v[k+1, i] += A[i, j] * v[k, j]
                grad2_new = grad2_li(z_next[i], s[k+1, i], beta[i])
                grad2_old = grad2_li(z[k, i], s[k, i], beta[i])
                v[k+1, i] += grad2_new - grad2_old
                
            z[k+1] = z_next

        cost, grad_norm, consensus, sigma_err = _compute_metrics(z, r_targets, gamma, beta, N, max_iters)
        
        # Calculate strict consensus error across the network
        for k in range(max_iters):
            sigma_k = np.mean(z[k], axis=0)
            consensus[k] = np.mean(np.linalg.norm(s[k] - sigma_k, axis=1))

        print(f"  [{graph_type.capitalize()}] Final cost: {cost[-1]:.4f} | Final |∇J|: {grad_norm[-1]:.2e} | Consensus: {consensus[-1]:.2e}")
        
        scenario = {
            "label": graph_type.capitalize(),
            "graph_type": graph_type,
            "N": N,
            "z_init": z_init,
            "r_targets": r_targets,
            "z_opt": z_opt,
            "sigma_opt": sigma_opt,
            "z_hist": z,
            "metrics": {
                "cost": cost,
                "grad_norm": grad_norm,
                "consensus": consensus,
                "sigma_error": sigma_err
            }
        }
        scenarios.append(scenario)

    return scenarios