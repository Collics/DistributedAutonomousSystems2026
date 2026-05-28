import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import Parameters as par
 
#  Task 1.1
def plot_task_1_1_summary_results(all_costs, all_grads, all_consensus):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    cmap = plt.get_cmap('Dark2')
    
    for idx, (gt, cost_hist) in enumerate(all_costs.items()):
        axes[0].plot(cost_hist, label=f"Topology: {gt}", linewidth=2.5, color=cmap(idx))
    axes[0].set_title(r"Evolution of the Global Cost Function $l(\bar{z})$", fontweight='bold')
    axes[0].set_ylabel("Cost Function")
    axes[0].legend()

    for idx, (gt, grad_hist) in enumerate(all_grads.items()):
        axes[1].semilogy(grad_hist, label=f"Topology: {gt}", linewidth=2.5, color=cmap(idx))
    axes[1].set_title(r"Evolution of the Global Gradient Norm $\|\nabla l(\bar{z})\|$ (Log Scale)", fontweight='bold')
    axes[1].set_ylabel("Norm of Gradient (Log)")
    axes[1].legend()

    for idx, (gt, consensus_hist) in enumerate(all_consensus.items()):
        axes[2].plot(consensus_hist, label=f"Topology: {gt}", linewidth=2.5, color=cmap(idx))
    axes[2].set_title(r"Evolution of the Consensus Error $\|z - \bar{z}\|$", fontweight='bold')
    axes[2].set_ylabel("Consensus Error")
    axes[2].set_xlabel("Iterations k") 
    axes[2].legend()

    plt.tight_layout(h_pad=2.0)
    plt.show()

def plot_task_1_1_network(all_graphs):
    num_graphs = len(all_graphs)
    fig, axes = plt.subplots(nrows=1, ncols=num_graphs, figsize=(6 * num_graphs, 5))
    if num_graphs == 1: axes = [axes]
        
    for ax, (gt, G) in zip(axes, all_graphs.items()):
        ax.set_title(f"Graph: {gt.upper()}", fontsize=13, fontweight='bold', color='#333333')
        pos = nx.spring_layout(G, seed=42) 
        # Cambiati forma (esagono), colori e spessori per renderlo unico
        nx.draw(G, pos, ax=ax, with_labels=True, 
                node_color='#F08080', edge_color='#888888', 
                node_size=800, node_shape='h', font_weight='bold', 
                font_color='white', width=1.5)
        ax.set_facecolor('#ffffff')
    plt.tight_layout()
    plt.show()

def plot_task_1_1_consensus_dynamics(z_history, topology_name, z_star=None):
    max_iters, n_agents, d = z_history.shape
    fig, axes = plt.subplots(nrows=d, ncols=1, figsize=(10, 4 * d), sharex=True)
    if d == 1: axes = [axes]
    cmap = plt.get_cmap('Set2')
        
    for comp in range(d):
        ax = axes[comp]
        if z_star is not None:
            # Linea dell'ottimo modificata
            ax.axhline(y=z_star[comp], color="#000000", linestyle='-.', linewidth=2.5, alpha=0.85, 
                       label=r'Optimal $z^*$ (Centralized)', zorder=2)
            
        for i in range(n_agents):
            ax.plot(z_history[:, i, comp], label=f"Agent {i+1}", linewidth=1.8, color=cmap(i), zorder=3)
            
        ax.set_title(f"Component $z[{comp}]$ Consensus - {topology_name.capitalize()}", fontweight='bold')
        ax.set_ylabel(f"Value $z[{comp}]$")
        
        if comp == 0:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    axes[-1].set_xlabel("Iteration k")
    plt.tight_layout()
    plt.show()



 

#  Task 1.2

def plot_task_1_2_datasets(X, labels_cubic, labels_super, phi_c=None, phi_s=None, w_c=None, b_c=None, w_s=None, b_s=None, title_prefix=""):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True)
    configs = [
        {"name": "Parabola", "labels": labels_cubic, "phi": phi_c, "w": w_c, "b": b_c},
        {"name": "Hyperbola", "labels": labels_super, "phi": phi_s, "w": w_s, "b": b_s}
    ]
    
    x_range = np.linspace(par.TASK_1_2_RANGE[0], par.TASK_1_2_RANGE[1], 100)
    y_range = np.linspace(par.TASK_1_2_RANGE[0], par.TASK_1_2_RANGE[1], 100)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)

    for ax, conf in zip(axes, configs):
        ax.scatter(X[conf["labels"] == 1, 0], X[conf["labels"] == 1, 1], 
                   color='#8A2BE2', marker='^', label='+1 (Healthy)', alpha=0.6, s=25, edgecolors='none')
        ax.scatter(X[conf["labels"] == -1, 0], X[conf["labels"] == -1, 1], 
                   color='#FF8C00', marker='v', label='-1 (Defective)', alpha=0.6, s=25, edgecolors='none')
        
        if conf["w"] is not None and conf["b"] is not None and conf["phi"] is not None:
            Z = np.zeros(X_mesh.shape)
            for i in range(X_mesh.shape[0]):
                for j in range(X_mesh.shape[1]):
                    point = np.array([X_mesh[i,j], Y_mesh[i,j]])
                    Z[i,j] = np.dot(conf["w"], conf["phi"](point)) + conf["b"]
            
            ax.contour(X_mesh, Y_mesh, Z, levels=[0], colors='#2F4F4F', linestyles='-.', linewidths=2.5)
            ax.plot([], [], color='#2F4F4F', linestyle='-.', linewidth=2.5, label='Decision Boundary')

        ax.set_title(f"{title_prefix}{conf['name']} Feature Mapping", fontweight='bold')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect('equal')
        ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.show()

def plot_task_1_2_metrics(cost_hist, grad_hist, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(cost_hist, color='#20B2AA', linewidth=2.5) 
    axes[0].set_title(f"Centralized Loss Evolution - {title}", fontweight='bold')
    axes[0].set_ylabel("Loss Function")
    
    axes[1].semilogy(grad_hist, color='#C71585', linewidth=2.5) 
    axes[1].set_title(f"Gradient Norm (Log) - {title}", fontweight='bold')
    axes[1].set_ylabel("Norm")
    axes[1].set_xlabel("Iteration k")
    
    plt.tight_layout()
    plt.show()

def plot_task1_2_dataset(X, labels, phi_fn=None, wb=None, map_name="", data_range=(-3, 3)):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Task 1.2 – Data & Boundary ({map_name})", fontsize=13, fontweight="bold")
 
    ax.scatter(X[labels ==  1, 0], X[labels ==  1, 1],
               color="#8A2BE2", marker='^', label="Class +1", alpha=0.6, s=25, edgecolors='none')
    ax.scatter(X[labels == -1, 0], X[labels == -1, 1],
               color="#FF8C00", marker='v', label="Class −1", alpha=0.6, s=25, edgecolors='none')
 
    if phi_fn is not None and wb is not None:
        grid = np.linspace(data_range[0], data_range[1], 200)
        Xm, Ym = np.meshgrid(grid, grid)
        pts = np.column_stack([Xm.ravel(), Ym.ravel()])
        Phi_grid = phi_fn(pts)
        Z = (Phi_grid @ wb[:-1] + wb[-1]).reshape(Xm.shape)
        ax.contour(Xm, Ym, Z, levels=[0], colors="#2F4F4F", linestyles='-.', linewidths=2.5)
        ax.plot([], [], color="#2F4F4F", linestyle='-.', linewidth=2.5, label="Boundary")
 
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    return fig

#  Task 1.3
 
def plot_task_1_3_individual_distr_metrics(centr_cost, centr_grad, dist_cost, dist_grad, g_type, mapping_name):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    
    axes[0].plot(centr_cost, color='#708090', linestyle=':', linewidth=2, label="Centr. (Ideal)", alpha=0.8)
    axes[0].plot(dist_cost, color='#4169E1', linewidth=2, label=f"Distr. ({g_type.upper()})")
    axes[0].set_title(f"Cost Comparison: {mapping_name} [{g_type.upper()}]", fontweight='bold')
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    axes[1].semilogy(centr_grad, color='#708090', linestyle=':', linewidth=2, label="Centr. (Ideal)", alpha=0.8)
    axes[1].semilogy(dist_grad, color='#4169E1', linewidth=2, label=f"Distr. ({g_type.upper()})")
    axes[1].set_title("Gradient Norm (Log)", fontweight='bold')
    axes[1].set_ylabel("Norm")
    axes[1].set_xlabel("Iteration k")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_task_1_3_metrics(centr_cost, cent_grad, dist_costs, dist_grads, title_prefix):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    cmap = plt.get_cmap('Dark2')

    axes[0].plot(centr_cost, color='#708090', linestyle=':', linewidth=2, alpha=0.8, label="Centr. Baseline", zorder=2)
    for i, (g_type, cost_hist) in enumerate(dist_costs.items()):
        axes[0].plot(cost_hist, label=f"Distr. {g_type.upper()}", color=cmap(i), linewidth=1.8, zorder=3)
    axes[0].set_title(f"Global Cost - {title_prefix}", fontweight='bold')
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    axes[1].semilogy(cent_grad, color='#708090', linestyle=':', linewidth=2, alpha=0.8, label="Centr. Baseline", zorder=2)
    for i, (g_type, grad_hist) in enumerate(dist_grads.items()):
        axes[1].semilogy(grad_hist, label=f"Distr. {g_type.upper()}", color=cmap(i), linewidth=1.8, zorder=3)
    axes[1].set_title(f"Global Gradient Norm - {title_prefix}", fontweight='bold')
    axes[1].set_ylabel("Norm (Log)")
    axes[1].set_xlabel("Iteration k")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_task1_3_data_split(agents_data, title_prefix="Task 1.3"):
    cmap = plt.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(9, 7))
    
    fig.suptitle(f"{title_prefix} – Agent Dataset Allocation", fontsize=14, fontweight="bold")
 
    for i, data in enumerate(agents_data):
        ax.scatter(data[:, 0], data[:, 1], color=cmap(i), marker='p', label=f"Node {i + 1}", alpha=0.75, edgecolors="white", s=45)
 
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
 
    fig.subplots_adjust(right=0.78)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, framealpha=0.9)
 
    plt.tight_layout()
    plt.show() 
    return fig


# ──────────────────────────────────────────────────────────────
#  Task 2.1
# ──────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =============================================================================
# TASK 2.1 – Plot 1: Convergence metrics for a single scenario
# =============================================================================

def plot_task2_1_metrics(scenario):
    """
    Plots the evolution of the global cost function, the gradient norm,
    the σ estimation error and the consensus error for one scenario.

    Parameters
    ----------
    scenario : dict returned by run_task2_1()
    """
    m   = scenario["metrics"]
    lbl = scenario.get("label", "")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
    fig.suptitle(
        rf"Task 2.1 – Convergence Metrics  |  {lbl}",
        fontsize=13, fontweight="bold",
    )

    specs = [
        (axes[0, 0], m["cost"],        r"Global Cost $J(z, \sigma)$",          False),
        (axes[0, 1], m["grad_norm"],   r"Gradient Norm $\|\nabla J\|$",         True),
        (axes[1, 0], m["sigma_error"], r"$\sigma$ Estimation Error (Log Scale)", True),
        (axes[1, 1], m["consensus"],   r"Consensus Error $\|z - \bar{z}\|$",     True),
    ]

    for ax, data, title, use_log in specs:
        if use_log:
            ax.semilogy(data, linewidth=2)
        else:
            ax.plot(data, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Iteration $k$")
        ax.grid(True, which="both", alpha=0.4)

    plt.tight_layout(h_pad=2.5)
    plt.show()


# =============================================================================
# TASK 2.1 – Plot 2: 2-D robot trajectories
# =============================================================================

def plot_task2_1_trajectories(scenario, subsample=5):
    """
    2-D visualisation of robot paths, private targets, barycenter trajectory
    and final optimal positions.

    Parameters
    ----------
    scenario  : dict returned by run_task2_1()
    subsample : plot every k-th step along each trajectory (for performance)
    """
    z_hist    = scenario["z_hist"]      # (K, N, 2)
    r_targets = scenario["r_targets"]   # (N, 2)
    z_init    = scenario["z_init"]      # (N, 2)
    z_opt     = scenario["z_opt"]       # (N, 2)
    sigma_opt = scenario["sigma_opt"]   # (2,)
    N         = scenario["N"]
    lbl       = scenario.get("label", "")

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i / max(N - 1, 1)) for i in range(N)]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(
        rf"Task 2.1 – Robot Trajectories  |  {lbl}",
        fontsize=13, fontweight="bold",
    )

    # ── Barycenter trajectory ─────────────────────────────────────────────
    sigma_hist = z_hist.mean(axis=1)   # (K, 2)
    ax.plot(
        sigma_hist[::subsample, 0], sigma_hist[::subsample, 1],
        color="green", linewidth=2.0, linestyle="--", zorder=3,
        label="Barycenter path",
    )
    ax.scatter(*sigma_hist[0],  marker="^", s=100, color="green",
               zorder=5, edgecolors="k", linewidths=0.6)
    ax.scatter(*sigma_opt, marker="P", s=200, color="green",
               zorder=6, edgecolors="k", linewidths=0.8,
               label=r"$\sigma^*$ (optimal barycenter)")

    # ── Per-robot trajectories ────────────────────────────────────────────
    for i in range(N):
        c    = colors[i]
        traj = z_hist[::subsample, i, :]
        ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=1.3,
                alpha=0.7, zorder=2)
        # Start
        ax.scatter(*z_init[i], marker="o", s=70, color=c,
                   zorder=4, edgecolors="k", linewidths=0.5)
        # End
        ax.scatter(*z_hist[-1, i], marker="s", s=80, color=c,
                   zorder=4, edgecolors="k", linewidths=0.5)
        # Optimal position
        ax.scatter(*z_opt[i], marker="*", s=160, color=c,
                   zorder=5, edgecolors="k", linewidths=0.5)
        # Private target
        ax.scatter(*r_targets[i], marker="x", s=100,
                   color=c, zorder=3, linewidths=2.0)
        # Label
        ax.annotate(f"R{i}", z_init[i], fontsize=7.5, color=c,
                    xytext=(4, 4), textcoords="offset points")

    # ── Dummy handles for the legend ─────────────────────────────────────
    ax.scatter([], [], marker="o", s=60, c="gray",  label="Start")
    ax.scatter([], [], marker="s", s=60, c="gray",  label="Final position")
    ax.scatter([], [], marker="*", s=120, c="gray", label=r"Optimal $z_i^*$")
    ax.scatter([], [], marker="x", s=80,  c="gray", label=r"Private target $r_i$")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Put legend outside to avoid cluttering the trajectory plot
    fig.subplots_adjust(right=0.75)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.show()


# =============================================================================
# TASK 2.1 – Plot 3: Metric comparison across multiple scenarios
# =============================================================================

def plot_task2_1_comparison(scenarios):
    """
    Overlays the convergence metrics of multiple scenarios in one figure,
    following the same style as plot_combined_results() in plot_utils.py.

    Parameters
    ----------
    scenarios : list of dicts returned by run_task2_1()
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "Task 2.1 – Scenario Comparison",
        fontsize=13, fontweight="bold",
    )

    specs = [
        (axes[0, 0], "cost",        r"Global Cost $J(z, \sigma)$",           False),
        (axes[0, 1], "grad_norm",   r"Gradient Norm $\|\nabla J\|$ (Log)",    True),
        (axes[1, 0], "sigma_error", r"$\sigma$ Estimation Error (Log)",        True),
        (axes[1, 1], "consensus",   r"Consensus Error $\|z - \bar{z}\|$ (Log)", True),
    ]

    for ax, key, title, use_log in specs:
        for sc in scenarios:
            lbl  = sc.get("label", "")
            data = sc["metrics"][key]
            if use_log:
                ax.semilogy(data, linewidth=2, label=f"Topology: {lbl}")
            else:
                ax.plot(data,    linewidth=2, label=f"Topology: {lbl}")
        ax.set_title(title)
        ax.set_xlabel("Iteration $k$")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()

    plt.tight_layout(h_pad=2.5)
    plt.show()


# =============================================================================
# TASK 2.1 – Plot 4: Animated 2-D visualisation
# =============================================================================

def animate_task2_1(scenario, skip_frames=5, save_mp4=False):
    """
    Animates the team behaviour during the aggregative optimisation.
    Style mirrors animate_team_behavior() in animation_utils.py.

    Parameters
    ----------
    scenario    : dict returned by run_task2_1()
    skip_frames : render every k-th iteration (speeds up the animation)
    save_mp4    : if True, saves the animation as 'task2_1_animation.mp4'
                  (requires ffmpeg)
    """
    z_hist    = scenario["z_hist"]      # (K, N, 2)
    r_targets = scenario["r_targets"]   # (N, 2)
    z_opt     = scenario["z_opt"]       # (N, 2)
    N         = scenario["N"]
    lbl       = scenario.get("label", "")
    maxK      = z_hist.shape[0]

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i / max(N - 1, 1)) for i in range(N)]

    fig, ax = plt.subplots(figsize=(8, 8))

    # ── Dynamic axis limits ───────────────────────────────────────────────
    all_x = np.concatenate([z_hist[:, :, 0].flatten(), r_targets[:, 0]])
    all_y = np.concatenate([z_hist[:, :, 1].flatten(), r_targets[:, 1]])
    padding = 2.0
    ax.set_xlim(all_x.min() - padding, all_x.max() + padding)
    ax.set_ylim(all_y.min() - padding, all_y.max() + padding)

    ax.set_title(rf"Task 2.1 – Aggregative Tracking Animation  |  {lbl}")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.grid(True)
    ax.set_aspect("equal")

    # ── Static elements ───────────────────────────────────────────────────
    ax.scatter(r_targets[:, 0], r_targets[:, 1],
               marker="x", s=100, color="red", linewidths=2,
               label=r"Private targets $r_i$")
    ax.scatter(z_opt[:, 0], z_opt[:, 1],
               marker="*", s=150, color="orange", edgecolors="k",
               linewidths=0.6, zorder=3,
               label=r"Optimal positions $z_i^*$")
    ax.scatter(z_hist[0, :, 0], z_hist[0, :, 1],
               marker="o", s=60, color="blue", alpha=0.3, label="Start")

    # ── Dynamic elements ─────────────────────────────────────────────────
    robots_scatter     = ax.scatter([], [], marker="o", s=80, color="blue",
                                    label="Robots", zorder=5)
    barycenter_scatter = ax.scatter([], [], marker="D", s=120, color="green",
                                    label=r"Barycenter $\sigma(z)$", zorder=6)
    tails = [ax.plot([], [], linestyle="--", color=colors[i], alpha=0.5)[0]
             for i in range(N)]

    # Legend outside the plot (same pattern as animation_utils.py)
    fig.subplots_adjust(right=0.75)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    # ── Init / update callbacks ───────────────────────────────────────────
    def init():
        robots_scatter.set_offsets(np.empty((0, 2)))
        barycenter_scatter.set_offsets(np.empty((0, 2)))
        for tail in tails:
            tail.set_data([], [])
        return [robots_scatter, barycenter_scatter] + tails

    def update(frame):
        current_z = z_hist[frame]
        robots_scatter.set_offsets(current_z)

        sigma_z = np.mean(current_z, axis=0)
        barycenter_scatter.set_offsets(sigma_z)

        tail_length = 30
        start_idx   = max(0, frame - tail_length)
        for i in range(N):
            tails[i].set_data(
                z_hist[start_idx:frame + 1, i, 0],
                z_hist[start_idx:frame + 1, i, 1],
            )
        return [robots_scatter, barycenter_scatter] + tails

    # ── Frame selection ───────────────────────────────────────────────────
    frames_to_render = np.arange(0, maxK, skip_frames)
    if frames_to_render[-1] != maxK - 1:
        frames_to_render = np.append(frames_to_render, maxK - 1)

    print("Generating animation… (this might take a few seconds)")
    anim = animation.FuncAnimation(
        fig, update, frames=frames_to_render,
        init_func=init, blit=True, interval=50,
    )

    if save_mp4:
        try:
            anim.save("task2_1_animation.mp4", fps=30,
                      extra_args=["-vcodec", "libx264"])
            print("Animation saved to 'task2_1_animation.mp4'.")
        except Exception as e:
            print(f"Error saving animation (ffmpeg might be missing): {e}")

    plt.show()