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
    return gamma_i * np.dot(zi - r_i, zi - r_i) + beta_i * np.dot(zi - sigma, zi - sigma)

def grad1_li(zi, sigma, r_i, gamma_i, beta_i):
    return 2.0 * gamma_i * (zi - r_i) + 2.0 * beta_i * (zi - sigma)

def grad2_li(zi, sigma, beta_i):
    return -2.0 * beta_i * (zi - sigma)

def generate_safe_initial_positions(N, box_size=5.0, min_dist=0.5, offset_x=0.0, offset_y=0.0):
    z0 = np.zeros((N, 2))
    for i in range(N):
        while True:
            candidate = np.random.uniform(0, box_size, 2) + np.array([offset_x, offset_y])
            if i == 0:
                z0[i] = candidate
                break
            distances = np.linalg.norm(z0[:i] - candidate, axis=1)
            if np.all(distances >= min_dist):
                z0[i] = candidate
                break
    return z0

def generate_target_geometry(N, shape='hexagon', scale=3.0, center=[6.0, 6.0]):
    targets = np.zeros((N, 2))
    center = np.array(center)
    
    if shape == 'hexagon':
        for i in range(N):
            angle = i * (2 * np.pi / N)
            targets[i] = center + scale * np.array([np.cos(angle), np.sin(angle)])
    elif shape == 'line':
        start_x = center[0] - scale
        end_x = center[0] + scale
        x_vals = np.linspace(start_x, end_x, N)
        for i in range(N):
            targets[i] = np.array([x_vals[i], center[1]])
    elif shape == 'triangle':
        v1 = center + scale * np.array([0, 1.0])
        v2 = center + scale * np.array([-np.sqrt(3)/2, -0.5])
        v3 = center + scale * np.array([np.sqrt(3)/2, -0.5])
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        for i in range(N):
            edge_idx = i % 3
            step = (i // 3) / max(1, (N // 3))
            start_v, end_v = edges[edge_idx]
            targets[i] = start_v * (1 - step) + end_v * step
    else:
        for i in range(N):
            angle = i * (2 * np.pi / N)
            targets[i] = center + scale * np.array([np.cos(angle), np.sin(angle)])
    return targets

def _compute_metrics(z, r_targets, gamma, beta, N, max_iters):
    cost = np.zeros(max_iters)
    grad_norm = np.zeros(max_iters)
    consensus = np.zeros(max_iters)
    sigma_err = np.zeros(max_iters)

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

def _run_single_scenario(N, max_iters, stepsize, gamma_val, beta_val, target_shape, graph_type, z_init, label):
    """Executes the algorithm for a specific set of parameters and returns the scenario dict."""
    print(f"  [Running] {label} (γ={gamma_val}, λ={beta_val}, shape={target_shape})")
    
    r_targets = generate_target_geometry(N, shape=target_shape, scale=3.0, center=[6.0, 6.0])
    gamma = np.ones(N) * gamma_val
    beta = np.ones(N) * beta_val

    num = np.zeros(2)
    den_sub = 0.0
    for i in range(N):
        num += (gamma[i] * r_targets[i]) / (gamma[i] + beta[i])
        den_sub += beta[i] / (gamma[i] + beta[i])
    sigma_opt = (num / N) / (1.0 - den_sub / N)

    z_opt = np.zeros((N, 2))
    for i in range(N):
        z_opt[i] = (gamma[i] * r_targets[i] + beta[i] * sigma_opt) / (gamma[i] + beta[i])

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
        for i in range(N):
            g1 = grad1_li(z[k, i], s[k, i], r_targets[i], gamma[i], beta[i])
            g_phi = grad_phi(z[k, i])
            u_nom = -stepsize * (g1 + g_phi * v[k, i])
            z_next[i] = z[k, i] + u_nom
            
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
    
    for k in range(max_iters):
        sigma_k = np.mean(z[k], axis=0)
        consensus[k] = np.mean(np.linalg.norm(s[k] - sigma_k, axis=1))

    return {
        "label": label,
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
def run_task2_1():
    print("======================================================================")
    print("--- Starting Task 2.1: Automated Aggregative Tracking Tests ---")
    print("======================================================================")
    
    N = getattr(par, 'TASK_2_1_N', 6)
    max_iters = getattr(par, 'TASK_2_1_MAX_ITER', 15000)
    stepsize = getattr(par, 'TASK_2_1_ALPHA', 0.01)

    # Start Positions: Randomly in a box (safe from overlapping)
    np.random.seed(0) 
    z_init = generate_safe_initial_positions(N, box_size=5.0, min_dist=0.5, offset_x=-10.0)

    opt_gamma, opt_lambda = getattr(par, 'TASK_2_1_PARAM', (1.0, 1.0))
    base_shape = 'hexagon'
    base_graph = 'cycle'

    # ---------------------------------------------------------
    # EXPERIMENT 1: Tuning Parameters
    # ---------------------------------------------------------
    print("\n[Experiment 1] Testing Tuning Parameters...")
    param_sets = getattr(par, 'TASK_2_1_PARAM_SETS', [(1.0, 1.0), (0.1, 2.0), (2.0, 0.1)])
    param_labels = getattr(par, 'TASK_2_1_PARAM_LABELS', ["Balanced", "Cohesion", "Target Drive"])
    
    scenarios_params = []
    for (g_val, b_val), label in zip(param_sets, param_labels):
        sc = _run_single_scenario(N, max_iters, stepsize, g_val, b_val, base_shape, base_graph, z_init, label)
        scenarios_params.append(sc)

    # ---------------------------------------------------------
    # EXPERIMENT 2: Target Locations
    # ---------------------------------------------------------
    print("\n[Experiment 2] Testing Target Geometries...")
    shapes = getattr(par, 'TASK_2_1_SHAPES', ['hexagon', 'triangle', 'line'])
    
    scenarios_geom = []
    for shape in shapes:
        sc = _run_single_scenario(N, max_iters, stepsize, opt_gamma, opt_lambda, shape, base_graph, z_init, shape.capitalize())
        scenarios_geom.append(sc)

    # ---------------------------------------------------------
    # EXPERIMENT 3: Network Topologies
    # ---------------------------------------------------------
    print("\n[Experiment 3] Testing Network Topologies...")
    graphs = getattr(par, 'TASK_2_1_GRAPHS', ['cycle', 'path', 'star'])
    
    scenarios_graphs = []
    for g in graphs:
        sc = _run_single_scenario(N, max_iters, stepsize, opt_gamma, opt_lambda, base_shape, g, z_init, f"{g.capitalize()} Graph")
        scenarios_graphs.append(sc)

    # Return grouped dictionaries
    return {
        "Experiment 1: Parameter Tuning\n(Fixed: Hexagon Target, Cycle Graph)": scenarios_params,
        "Experiment 2: Target Geometries\n(Fixed: Balanced Params, Cycle Graph)": scenarios_geom,
        "Experiment 3: Network Topologies\n(Fixed: Balanced Params, Hexagon Target)": scenarios_graphs
    }