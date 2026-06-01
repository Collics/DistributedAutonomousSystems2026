"""
Distributed Autonomous Systems 
Task 2.3: Multi-Robot Safety Controllers using CBF-QP
Authors: Ivan Colangelo, Nicholas Gioia, Alexandru Zaporojanu
Bologna, 09/06/26
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint

import Parameters as par
from graph_utils import get_graph_and_matrix

# Import core tracking math
from tasks.task2_1_DEF import grad1_li, grad2_li, phi, grad_phi, _compute_metrics

# Import cleanly separated plotting utilities!
from plots import (
    plot_task2_3_trajectories,
    plot_task2_3_metrics,
    plot_task2_3_safety,
    plot_task2_3_animation
)

def cbf_qp_filter(z_i, u_nom, obstacles, d_safe, gamma_cbf):
    """Solve the 2D CBF-QP using scipy.optimize.minimize."""
    if obstacles.size == 0:
        return u_nom, False

    diff = z_i[None, :] - obstacles
    grad_V = 2.0 * diff
    A = -grad_V
    b = gamma_cbf * (np.sum(diff**2, axis=1) - d_safe**2)

    def objective(u):
        return np.sum((u - u_nom)**2)
    
    def jacobian(u):
        return 2 * (u - u_nom)

    constraints = LinearConstraint(A, -np.inf, b)
    u0 = np.copy(u_nom)

    res = minimize(
        objective, u0, method='SLSQP', jac=jacobian, 
        constraints=constraints, options={'ftol': 1e-9, 'disp': False}
    )

    if res.success:
        best_u = res.x
        cost = float(np.sum((best_u - u_nom) ** 2))
        activated = cost > 1e-6
        return best_u, activated
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

def simulate_team(use_safety):
    """Runs the Aggregative Tracking algorithm with optional CBF-QP safety filter."""
    N         = par.TASK_2_3_N
    max_iters = par.TASK_2_3_MAX_ITER
    stepsize  = par.TASK_2_3_ALPHA
    gamma_val = par.TASK_2_3_GAMMA
    beta_val  = par.TASK_2_3_BETA
    gamma_cbf = par.TASK_2_3_GAMMA_CBF
    d_safe    = par.TASK_2_3_D_SAFE
    graph     = par.TASK_2_3_GRAPH
    obstacles = np.array(par.TASK_2_3_OBSTACLES, dtype=float)

    gamma = np.ones(N) * gamma_val
    beta  = np.ones(N) * beta_val
    
    init_center = np.array([-7.0, 0.0])
    init_radius = 1.0
    z_init = np.array([
        [init_center[0] + init_radius * np.cos(2.0 * np.pi * i / N),
         init_center[1] + init_radius * np.sin(2.0 * np.pi * i / N)]
        for i in range(N)
    ])

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
        'z': z,
        'u_nom': u_nom_hist,
        'u_app': u_app_hist,
        'min_clearance': min_clearance,
        'min_cbf_value': min_cbf_value,
        'cost_history': cost,
        'grad_norm_history': grad_norm,
        'safety_activations': safety_activations,
        'obstacles': obstacles,
        'd_safe': d_safe,
        'targets': r,
        'graph_type': graph,
        'gamma_cbf': gamma_cbf,
    }

def run_task_2_3():
    print("======================================================================")
    print("--- Starting Task 2.3: Multi-Robot Safety Controllers ---")
    print("======================================================================")

    nominal = simulate_team(use_safety=False)
    safe = simulate_team(use_safety=True)

    print(f"Optimization finished with graph topology: '{nominal['graph_type']}'")
    print(f"CBF safety parameter gamma: {safe['gamma_cbf']:.2f}")
    print(f"Minimum nominal clearance: {np.min(nominal['min_clearance']):.4f}")
    print(f"Minimum safe clearance: {np.min(safe['min_clearance']):.4f}")
    print(f"Safety activations: {safe['safety_activations']}")

    if getattr(par, 'TASK_2_3_PLOTS', False):
        # The plotting logic is now cleanly abstracted!
        fig1 = plot_task2_3_trajectories(nominal, safe)
        fig2 = plot_task2_3_metrics(nominal, safe)
        fig3 = plot_task2_3_safety(nominal, safe)

        if getattr(par, 'TASK_2_3_ANIMATION', False) or getattr(par, 'TASK_2_3_ANIMATE', False):
            print("Rendering animation...")
            global keep_alive_anim 
            keep_alive_anim, fig_anim = plot_task2_3_animation(nominal, safe, step=5, interval=40)
            
        plt.show()

    return nominal, safe # Standard practice to return data for analysis

if __name__ == "__main__":
    run_task_2_3()