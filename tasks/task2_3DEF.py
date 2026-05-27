"""
Distributed Autonomous Systems - Final Project
Task 2.3: Multi-Robot Safety Controllers using CBF-QP
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize, LinearConstraint

import Parameters as par
from graph_utils import get_graph_and_matrix

# Import the core tracking math and metrics from your existing framework
from tasks.task2_1_DEF import grad1_li, grad2_li, phi, grad_phi, _compute_metrics
from plots_task2 import plot_task2_3_animation

def cbf_qp_filter(z_i, u_nom, obstacles, d_safe, gamma_cbf):
    """Solve the 2D CBF-QP using scipy.optimize.minimize."""
    if obstacles.size == 0:
        return u_nom, False

    # Computation of the CBF constraint matrices A and b
    # Constraint: -∇V_o(z_k)^T u_k <= gamma * V_o(z_k) --> A * u <= b
    diff = z_i[None, :] - obstacles
    grad_V = 2.0 * diff
    A = -grad_V
    b = gamma_cbf * (np.sum(diff**2, axis=1) - d_safe**2)

    # Objective function for SciPy: ||u - u_nom||^2
    def objective(u):
        return np.sum((u - u_nom)**2)
    
    # Jacobian of the objective function
    def jacobian(u):
        return 2 * (u - u_nom)

    # Constraints A*u <= b
    constraints = LinearConstraint(A, -np.inf, b)

    u0 = np.copy(u_nom)

    # Execution of the optimization using SLSQP method
    res = minimize(
        objective, 
        u0, 
        method='SLSQP', 
        jac=jacobian, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )

    if res.success:
        best_u = res.x
        # Check if the filter has been activated significantly
        cost = float(np.sum((best_u - u_nom) ** 2))
        activated = cost > 1e-6
        return best_u, activated
    else:
        # If infeasible, apply emergency stop
        print(f"Warning: QP Infeasible or not converged. Applying emergency stop.")
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


def draw_obstacles(ax, obstacles, d_safe):
    """Draws obstacles on a Matplotlib axis."""
    for idx, center in enumerate(obstacles):
        circle = Circle(center, d_safe, facecolor='tab:orange', edgecolor='tab:red', alpha=0.25, linewidth=2)
        ax.add_patch(circle)
        ax.text(center[0] + 0.2, center[1] + 0.2, f'O{idx + 1}', color='tab:red', fontsize=10)


def simulate_team(use_safety):
    """
    Runs the Aggregative Tracking algorithm using existing logic from task2_1_DEF.py,
    optionally applying the CBF-QP safety filter.
    """
    N         = par.TASK_2_3_N
    max_iters = par.TASK_2_3_MAX_ITER
    stepsize  = par.TASK_2_3_ALPHA
    gamma_val = par.TASK_2_3_GAMMA
    beta_val  = par.TASK_2_3_BETA
    gamma_cbf = par.TASK_2_3_GAMMA_CBF
    d_safe    = par.TASK_2_3_D_SAFE
    graph     = par.TASK_2_3_GRAPH
    obstacles = np.array(par.TASK_2_3_OBSTACLES, dtype=float)

    # ── Re-create environment from task2_1_DEF ──
    gamma = np.ones(N) * gamma_val
    beta  = np.ones(N) * beta_val
    
    # ── MODIFIED SCENARIO ──
    # 1. Spread initial positions in a circle on the left side of the map
    init_center = np.array([-7.0, 0.0])
    init_radius = 1.0
    z_init = np.array([
        [init_center[0] + init_radius * np.cos(2.0 * np.pi * i / N),
         init_center[1] + init_radius * np.sin(2.0 * np.pi * i / N)]
        for i in range(N)
    ])

    # 2. Arrange private targets in a regular hexagon/circle on the right side of the map
    target_center = np.array([7.0, 0.0])
    target_radius = 4.0
    r = np.array([
        [target_center[0] + target_radius * np.cos(2.0 * np.pi * i / N),
         target_center[1] + target_radius * np.sin(2.0 * np.pi * i / N)]
        for i in range(N)
    ])

    # ── Initialize Tracking Variables ──
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

    # ── Main Tracking Loop ──
    for k in range(max_iters - 1):
        z_next = np.zeros((N, 2))
        
        # z-update
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

        # s-update & v-update (from task2_1_DEF)
        for i in range(N):
            neighbours = np.where(Adj[i])[0]
            
            # Consensus tracker
            s[k+1, i]  = A[i, i] * s[k, i]
            for j in neighbours:
                s[k+1, i] += A[i, j] * s[k, j]
            s[k+1, i] += phi(z_next[i]) - phi(z[k, i])

            # Gradient tracker
            v[k+1, i]  = A[i, i] * v[k, i]
            for j in neighbours:
                v[k+1, i] += A[i, j] * v[k, j]
            grad2_new   = grad2_li(z_next[i], s[k+1, i], beta[i])
            grad2_old   = grad2_li(z[k,   i], s[k,   i], beta[i])
            v[k+1, i]  += grad2_new - grad2_old

        z[k+1] = z_next

    # ── Retrieve Global Metrics via built-in function ──
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

    if par.TASK_2_3_PLOTS:
        # ==========================================
        # FIGURE 1: Trajectories (Nominal vs Safe)
        # ==========================================
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
        for idx, (data, title) in enumerate([
            (nominal, f"Nominal Tracking - {par.TASK_2_3_GRAPH}"),
            (safe, f"CBF Safe Tracking - {par.TASK_2_3_GRAPH}")
        ]):
            ax = axes1[idx]
            draw_obstacles(ax, data['obstacles'], data['d_safe'])
            for i in range(data['z'].shape[1]):
                ax.plot(data['z'][:, i, 0], data['z'][:, i, 1], alpha=0.7, label=f'Robot {i + 1}')
                ax.scatter(data['z'][0, i, 0], data['z'][0, i, 1], marker='o', color='blue', alpha=0.5)
                ax.scatter(data['targets'][i, 0], data['targets'][i, 1], marker='x', s=100, color='red')
            ax.set_title(title)
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axis('equal')
        fig1.tight_layout()

        # ==========================================
        # FIGURE 2: Cost and Gradient Norm Evolution
        # ==========================================
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        axes2[0].plot(nominal['cost_history'], label='Nominal', linewidth=2, color='tab:blue')
        axes2[0].plot(safe['cost_history'], label='Safe (CBF)', linewidth=2, linestyle='--', color='tab:red')
        axes2[0].set_title('Global Cost Evolution')
        axes2[0].set_xlabel('Iteration k')
        axes2[0].set_ylabel(r'$J(z,\sigma)$')
        axes2[0].grid(True, linestyle=':', alpha=0.6)
        axes2[0].legend()

        axes2[1].semilogy(nominal['grad_norm_history'], label='Nominal', linewidth=2, color='tab:orange')
        axes2[1].semilogy(safe['grad_norm_history'], label='Safe (CBF)', linewidth=2, linestyle='--', color='tab:red')
        axes2[1].set_title('Gradient Norm Evolution')
        axes2[1].set_xlabel('Iteration k')
        axes2[1].set_ylabel(r'$\| \nabla J(z,\sigma) \|$')
        axes2[1].grid(True, linestyle=':', alpha=0.6)
        axes2[1].legend()
        fig2.tight_layout()

        # ==========================================
        # FIGURE 3: Safety Metrics
        # ==========================================
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        axes3[0].plot(nominal['min_clearance'], label='Nominal', linewidth=2)
        axes3[0].plot(safe['min_clearance'], label='CBF-QP safe', linewidth=2)
        axes3[0].axhline(0.0, color='k', linestyle='--', linewidth=1)
        axes3[0].set_title('Minimum clearance to obstacle boundary')
        axes3[0].set_xlabel('Iteration k')
        axes3[0].set_ylabel(r'$\|z_i - p_o\| - d_{safe}$')
        axes3[0].grid(True, linestyle=':', alpha=0.6)
        axes3[0].legend()

        control_delta = np.linalg.norm(safe['u_app'] - nominal['u_nom'], axis=2)
        axes3[1].plot(np.max(control_delta, axis=1), color='tab:purple', linewidth=2)
        axes3[1].set_title('Maximum safety correction per iteration')
        axes3[1].set_xlabel('Iteration k')
        axes3[1].set_ylabel(r'$\max(\|u_i^{safe} - u_i^{nom}\|)$')
        axes3[1].grid(True, linestyle=':', alpha=0.6)
        fig3.tight_layout()

        
        if getattr(par, 'TASK_2_3_ANIMATION', False):
            print("Rendering animation...")
            global keep_alive_anim 
            keep_alive_anim, fig_anim = plot_task2_3_animation(nominal, safe, step=5, interval=40)
            
        plt.show()

if __name__ == "__main__":
    run_task_2_3()