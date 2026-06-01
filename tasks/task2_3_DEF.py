"""
Distributed Autonomous Systems 
Task 2.3: Multi-Robot Safety Controllers using CBF-QP
Authors: Ivan Colangelo, Nicholas Gioia, Alexandru Zaporojanu
Bologna, 09/06/26
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
import os

import Parameters as par
from graph_utils import get_graph_and_matrix

# Import core tracking math
from tasks.task2_1_DEF import grad1_li, grad2_li, phi, grad_phi, _compute_metrics, generate_safe_initial_positions

# Import cleanly separated plotting utilities!
from plots import (
    plot_task2_3_trajectories,
    plot_task2_3_metrics,
    plot_task2_3_safety,
    plot_task2_3_animation,
    plot_task2_3_unified
)

def cbf_qp_filter(z_i, u_nom, obstacles, d_safe, gamma_cbf):
    """Solve the 2D CBF-QP using scipy.optimize.minimize."""
    if obstacles.size == 0:
        return u_nom, False

    diff = z_i[None, :] - obstacles
    grad_V = 2.0 * diff
    A = -grad_V
    b = gamma_cbf * (np.sum(diff**2, axis=1) - d_safe**2)

    def objective(u): return np.sum((u - u_nom)**2)
    def jacobian(u): return 2 * (u - u_nom)

    constraints = LinearConstraint(A, -np.inf, b)
    u0 = np.copy(u_nom)

    res = minimize(
        objective, u0, method='SLSQP', jac=jacobian, 
        constraints=constraints, options={'ftol': 1e-9, 'disp': False}
    )

    if res.success:
        best_u = res.x
        cost = float(np.sum((best_u - u_nom) ** 2))
        return best_u, cost > 1e-6
    else:
        print("Warning: QP Infeasible. Applying emergency stop.")
        return np.zeros(2), True

def compute_safety_metrics(z, obstacles, d_safe):
    """Computes minimum clearance and minimum CBF value across all iterations."""
    min_clearance = np.zeros(z.shape[0])
    min_cbf_value = np.zeros(z.shape[0])

    for k in range(z.shape[0]):
        diff = z[k, :, None, :] - obstacles[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        min_clearance[k] = np.min(distances - d_safe)
        min_cbf_value[k] = np.min(distances**2 - d_safe**2)

    return min_clearance, min_cbf_value


def simulate_team(use_safety, gamma_val, beta_val, gamma_cbf, obstacles, graph, z_init):
    """Runs the Aggregative Tracking algorithm with explicit configuration arguments."""
    N         = par.TASK_2_3_N
    max_iters = par.TASK_2_3_MAX_ITER
    stepsize  = par.TASK_2_3_ALPHA
    d_safe    = par.TASK_2_3_D_SAFE
    obstacles = np.array(obstacles, dtype=float)

    gamma = np.ones(N) * gamma_val
    beta  = np.ones(N) * beta_val

    target_center = np.array([7.0, 0.0])
    target_radius = 4.0
    r = np.array([
        [target_center[0] + target_radius * np.cos(2.0 * np.pi * i / N),
         target_center[1] + target_radius * np.sin(2.0 * np.pi * i / N)]
        for i in range(N)
    ])

    G, A = get_graph_and_matrix(N, graph)
    Adj  = (A - np.eye(N)) > 0

    z = np.zeros((max_iters, N, 2))
    s = np.zeros((max_iters, N, 2))
    v = np.zeros((max_iters, N, 2))
    u_nom_hist = np.zeros((max_iters - 1, N, 2))
    u_app_hist = np.zeros((max_iters - 1, N, 2))

    # Apply the shared initial state
    z[0] = z_init.copy()
    for i in range(N):
        s[0, i] = phi(z[0, i])
        v[0, i] = grad2_li(z[0, i], s[0, i], beta[i])

    safety_activations = 0

    for k in range(max_iters - 1):
        z_next = np.zeros((N, 2))
        
        for i in range(N):
            g1 = grad1_li(z[k, i], s[k, i], r[i], gamma[i], beta[i])
            g_phi = grad_phi(z[k, i])
            u_nom = -stepsize * (g1 + g_phi * v[k, i])
            u_nom_hist[k, i] = u_nom

            if use_safety:
                u_app, activated = cbf_qp_filter(z[k, i], u_nom, obstacles, d_safe, gamma_cbf)
                safety_activations += int(activated)
            else:
                u_app, activated = u_nom, False

            u_app_hist[k, i] = u_app
            z_next[i] = z[k, i] + u_app

        for i in range(N):
            neighbours = np.where(Adj[i])[0]
            s[k+1, i]  = A[i, i] * s[k, i]
            for j in neighbours:
                s[k+1, i] += A[i, j] * s[k, j]
            s[k+1, i] += phi(z_next[i]) - phi(z[k, i])

            v[k+1, i]  = A[i, i] * v[k, i]
            for j in neighbours:
                v[k+1, i] += A[i, j] * v[k, j]
            grad2_new   = grad2_li(z_next[i], s[k+1, i], beta[i])
            grad2_old   = grad2_li(z[k,   i], s[k,   i], beta[i])
            v[k+1, i]  += grad2_new - grad2_old

        z[k+1] = z_next

    cost, grad_norm, consensus, sigma_err = _compute_metrics(z, r, gamma, beta, N, max_iters)
    min_clearance, min_cbf_value = compute_safety_metrics(z, obstacles, d_safe)

    return {
        'z': z, 'u_nom': u_nom_hist, 'u_app': u_app_hist,
        'min_clearance': min_clearance, 'min_cbf_value': min_cbf_value,
        'cost_history': cost, 'grad_norm_history': grad_norm,
        'safety_activations': safety_activations, 'obstacles': obstacles,
        'd_safe': d_safe, 'targets': r, 'graph_type': graph, 'gamma_cbf': gamma_cbf,
    }

def run_task_2_3():
    """Automates the execution and plotting of all Task 2.3 experimental scenarios."""
    out_dir = getattr(par, 'TASK_2_3_EXP_DIR', 'figs/Task2_3_Experiments')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Baseline/Fixed Parameters
    base_gamma, base_beta = getattr(par, 'TASK_2_3_PARAM', (2.0, 2.0))
    base_gcbf      = getattr(par, 'TASK_2_3_GCBF', 0.5)
    base_obstacles = getattr(par, 'TASK_2_3_OBSTACLES', [[0.0, 2.0], [2.0, -2.0]])
    base_graph     = getattr(par, 'TASK_2_3_GRAPH', 'cycle')
    
    # 2. Lock in a single initial position so all experiments are perfectly comparable
    z_init = generate_safe_initial_positions(par.TASK_2_3_N, box_size=2.0, min_dist=0.5, offset_x=-10.0)

    # 3. Define the unified scenario runner
    def _run_single_scenario(gamma, beta, gcbf, obstacles, graph_type, label, exp_prefix):
        print(f"\n---> Starting: {exp_prefix.replace('_', ' ')} ({label}) <---")
        
        nominal = simulate_team(False, gamma, beta, gcbf, obstacles, graph_type, z_init)
        safe    = simulate_team(True, gamma, beta, gcbf, obstacles, graph_type, z_init)
        
        base_filename = os.path.join(out_dir, f"{exp_prefix}_{label.replace(' ', '_')}")
        
        # --- NEW: Call the single unified dashboard ---
        fig_dash = plot_task2_3_unified(nominal, safe)

        desc_title = f"{exp_prefix.replace('_', ' ')}: {label}"
        desc_params = f"Topology: {safe['graph_type'].capitalize()}  |  $\gamma_{{cbf}}$: {safe['gamma_cbf']}  |  Obstacles: {len(safe['obstacles'])}"
        
        # Apply the descriptive title to the top of the dashboard
        fig_dash.suptitle(f"{desc_title}\n{desc_params}", fontsize=16, fontweight='bold')
        fig_dash.subplots_adjust(top=0.92)  # Make room for the title
            
        fig_dash.savefig(f"{base_filename}_dashboard.pdf", bbox_inches='tight')
        plt.close(fig_dash)
        # ----------------------------------------------
        
        if getattr(par, 'TASK_2_3_ANIMATE', False):
            print("Rendering animation... (This might take a moment)")
            anim, fig_anim = plot_task2_3_animation(nominal, safe, step=5, interval=40)
            fig_anim.suptitle(f"{desc_title}\n{desc_params}", fontsize=14, fontweight='bold')
            fig_anim.tight_layout()
            fig_anim.subplots_adjust(top=0.88)
            try:
                anim.save(f"{base_filename}.gif", writer='pillow', fps=20)
            except Exception as e:
                print(f"Failed to save animation: {e}")
            plt.close(fig_anim)
        
        print(f"[SUCCESS] Saved {label}")

    print("======================================================================")
    print(" STARTING TASK 2.3 AUTOMATED BATCH EXPERIMENTS")
    print("======================================================================")

    # ---------------------------------------------------------
    # EXPERIMENT 1: CBF Parameter Tuning
    # ---------------------------------------------------------
    print("\n[Experiment 1] Testing CBF Parameter Tuning...")
    exp1_gcbfs  = getattr(par, 'TASK_2_3_EXP1_GCBF', [0.1, 0.5, 2.0])
    exp1_labels = getattr(par, 'TASK_2_3_EXP1_LABELS', ["Conservative", "Balanced", "Aggressive"])
    for gcbf, label in zip(exp1_gcbfs, exp1_labels):
        _run_single_scenario(base_gamma, base_beta, gcbf, base_obstacles, base_graph, label, "Exp1_Parameter_Tuning")

    # ---------------------------------------------------------
    # EXPERIMENT 2: Obstacle Geometries
    # ---------------------------------------------------------
    print("\n[Experiment 2] Testing Obstacle Geometries...")
    exp2_obs    = getattr(par, 'TASK_2_3_EXP2_OBSTACLES', [])
    exp2_labels = getattr(par, 'TASK_2_3_EXP2_LABELS', [])
    for obs, label in zip(exp2_obs, exp2_labels):
        _run_single_scenario(base_gamma, base_beta, base_gcbf, obs, base_graph, label, "Exp2_Obstacle_Geometry")

    # ---------------------------------------------------------
    # EXPERIMENT 3: Network Topologies
    # ---------------------------------------------------------
    print("\n[Experiment 3] Testing Network Topologies...")
    exp3_graphs = getattr(par, 'TASK_2_3_EXP3_GRAPHS', ['cycle', 'path', 'star'])
    exp3_labels = getattr(par, 'TASK_2_3_EXP3_LABELS', ["Cycle", "Path", "Star"])
    for g, label in zip(exp3_graphs, exp3_labels):
        _run_single_scenario(base_gamma, base_beta, base_gcbf, base_obstacles, g, label, "Exp3_Network_Topology")

    print("\nALL 9 SAFETY EXPERIMENTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_task_2_3()