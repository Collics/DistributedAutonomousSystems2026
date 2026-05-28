import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
 
#  Task 1.1
 
def plot_task1_1(cost, gradient, consensus,
                 title="Task 1.1 – Distributed Gradient Tracking"):
    """
    Three-panel convergence plot for a single topology run.
 
    Parameters
    ----------
    cost      : array-like, shape (K,) – global cost per iteration
    gradient  : array-like, shape (K,) – global gradient norm per iteration
    consensus : array-like, shape (K,) – consensus error per iteration
    title     : str – figure suptitle (includes topology name when called
                from main)
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
 
    # Plot 1: Global Cost Function
    axes[0].plot(cost[:-1], linewidth=2)
    axes[0].set_title(r"Evolution of the Global Cost Function $\ell(\bar{z})$")
    axes[0].set_ylabel("Cost Function")
    axes[0].grid(True, alpha=0.4)
 
    # Plot 2: Gradient Norm (log scale – linear on log = exponential convergence)
    axes[1].semilogy(np.abs(gradient[:-1]), linewidth=2)
    axes[1].set_title(
        r"Evolution of the Global Gradient Norm $\|\nabla\ell(\bar{z})\|$ (Log Scale)"
    )
    axes[1].set_ylabel("Norm of Gradient (Log Scale)")
    axes[1].grid(True, which="both", alpha=0.4)
 
    # Plot 3: Consensus Error
    axes[2].semilogy(consensus[:-1], linewidth=2)
    axes[2].set_title(r"Evolution of the Consensus Error $\|z - \bar{z}\|$")
    axes[2].set_ylabel("Consensus Error (Log Scale)")
    axes[2].set_xlabel("Iteration $k$")
    axes[2].grid(True, which="both", alpha=0.4)
 
    plt.tight_layout(h_pad=2.0)
    return fig
 
 
def plot_task1_1_comparison(results):
    """
    Overlays cost, gradient norm and consensus error for multiple topologies
    in a single figure, following the style of plot_combined_results() in
    plot_utils.py.
 
    Parameters
    ----------
    results : dict  {graph_type: {"cost": …, "gradient": …, "consensus": …}}
              as returned by task1_1() when called for multiple topologies.
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        "Task 1.1 – Topology Comparison", fontsize=13, fontweight="bold"
    )
 
    for g_type, m in results.items():
        lbl = f"Topology: {g_type.capitalize()}"
        axes[0].plot(m["cost"][:-1],              label=lbl, linewidth=2)
        axes[1].semilogy(np.abs(m["gradient"][:-1]), label=lbl, linewidth=2)
        axes[2].semilogy(m["consensus"][:-1],     label=lbl, linewidth=2)
 
    specs = [
        (axes[0], r"Evolution of the Global Cost Function $\ell(\bar{z})$",
         "Cost Function", False),
        (axes[1],
         r"Evolution of the Global Gradient Norm $\|\nabla\ell(\bar{z})\|$ (Log Scale)",
         "Norm of Gradient (Log Scale)", True),
        (axes[2], r"Evolution of the Consensus Error $\|z - \bar{z}\|$ (Log Scale)",
         "Consensus Error (Log Scale)", True),
    ]
    for ax, title, ylabel, _ in specs:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()
 
    axes[-1].set_xlabel("Iteration $k$")
    plt.tight_layout(h_pad=2.0)
    return fig
 
 
def plot_task1_1_network(G, graph_type):
    """
    Draws the network topology using a spring layout.
 
    Parameters
    ----------
    G          : networkx.Graph
    graph_type : str – topology name used as title
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f"Topology: {graph_type.capitalize()}",
                 fontsize=14, fontweight="bold")
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True,
            node_color="lightblue", edge_color="gray",
            node_size=600, font_weight="bold")
    plt.tight_layout()
    return fig
 
 

#  Task 1.2

def plot_task1_2(cost_hist, grad_norm_hist, map_name=""):
    """
    Two-panel training-curve plot for centralised logistic regression.
 
    Parameters
    ----------
    cost_hist      : list of floats – loss per iteration
    grad_norm_hist : list of floats – gradient norm per iteration
    map_name       : str – feature mapping name, e.g. 'Parabola'
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Task 1.2 – Centralised Logistic Regression ({map_name})",
        fontsize=13, fontweight="bold",
    )
 
    # Plot 1: Cost Function Evolution
    axes[0].plot(cost_hist, linewidth=2)
    axes[0].set_title(r"Evolution of the Loss $\mathcal{L}(w, b)$")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.4)
 
    # Plot 2: Gradient Norm (log scale)
    axes[1].semilogy(grad_norm_hist, linewidth=2)
    axes[1].set_title(
        r"Evolution of the Gradient Norm $\|\nabla\mathcal{L}\|$ (Log Scale)"
    )
    axes[1].set_ylabel(r"$\|\nabla\mathcal{L}\|$ (Log Scale)")
    axes[1].set_xlabel("Iteration $k$")
    axes[1].grid(True, which="both", alpha=0.4)
 
    plt.tight_layout(h_pad=2.0)
    return fig
 
 
def plot_task1_2_dataset(X, labels, phi_fn=None, wb=None,
                         map_name="", data_range=(-3, 3)):
    """
    Scatter plot of the 2-D dataset with optional decision boundary.
 
    Parameters
    ----------
    X         : (M, 2) dataset
    labels    : (M,)   in {-1, +1}
    phi_fn    : callable  x → ϕ(x)  (required to draw the boundary)
    wb        : (q+1,) weight vector [w; b] (required to draw the boundary)
    map_name  : str – used in the title
    data_range: (min, max) for the contour grid
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        f"Task 1.2 – Dataset & Decision Boundary ({map_name})",
        fontsize=13, fontweight="bold",
    )
 
    ax.scatter(X[labels ==  1, 0], X[labels ==  1, 1],
               color="blue", label="Class +1", alpha=0.5, s=15)
    ax.scatter(X[labels == -1, 0], X[labels == -1, 1],
               color="red",  label="Class −1", alpha=0.5, s=15)
 
    if phi_fn is not None and wb is not None:
        grid = np.linspace(data_range[0], data_range[1], 200)
        Xm, Ym = np.meshgrid(grid, grid)
        pts = np.column_stack([Xm.ravel(), Ym.ravel()])
        Phi_grid = phi_fn(pts)
        Z = (Phi_grid @ wb[:-1] + wb[-1]).reshape(Xm.shape)
        ax.contour(Xm, Ym, Z, levels=[0], colors="black", linewidths=2)
        ax.plot([], [], color="black", linewidth=2, label="Decision Boundary")
 
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────
#  Task 1.3
# ──────────────────────────────────────────────────────────────
 
def plot_task1_3(cost_history, grad_norm_history, consensus_history, title=""):
    """
    Three-panel convergence plot for one distributed run (single topology
    and dataset size).
 
    Parameters
    ----------
    cost_history      : list of floats
    grad_norm_history : list of floats
    consensus_history : list of floats
    title             : str – figure suptitle (includes topology / M)
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        title or "Task 1.3 – Distributed Logistic Regression",
        fontsize=13, fontweight="bold",
    )
 
    # Plot 1: Total Cost
    axes[0].plot(cost_history, linewidth=2)
    axes[0].set_title(r"Evolution of the Global Cost Function $\mathcal{L}$")
    axes[0].set_ylabel("Total Loss")
    axes[0].grid(True, alpha=0.4)
 
    # Plot 2: Gradient Norm (log scale)
    axes[1].semilogy(grad_norm_history, linewidth=2)
    axes[1].set_title(
        r"Evolution of the Gradient Norm $\|\nabla\mathcal{L}\|$ (Log Scale)"
    )
    axes[1].set_ylabel("Norm (Log Scale)")
    axes[1].grid(True, which="both", alpha=0.4)
 
    # Plot 3: Consensus Error (log scale)
    axes[2].semilogy(consensus_history, linewidth=2)
    axes[2].set_title(
        r"Evolution of the Consensus Error $\|z - \bar{z}\|$ (Log Scale)"
    )
    axes[2].set_ylabel("Consensus Error (Log Scale)")
    axes[2].set_xlabel("Iteration $k$")
    axes[2].grid(True, which="both", alpha=0.4)
 
    plt.tight_layout(h_pad=2.0)
    return fig
 
 
def plot_task1_3_comparison(results, centr_cost=None, centr_grad=None):
    """
    Overlays cost, gradient norm and consensus error across all topology ×
    dataset-size combinations from task1_3(), with an optional centralized
    baseline, following the style of plot_task1_3_metrics() in plot_utils.py.
 
    Parameters
    ----------
    results    : list of dicts returned by task1_3()  (keys: graph_name, M,
                 cost_history, grad_norm_history, consensus_history, title)
    centr_cost : list of floats (optional) – centralized baseline cost
    centr_grad : list of floats (optional) – centralized baseline grad norm
    """
    cmap = plt.get_cmap("tab10")
 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        "Task 1.3 – Distributed vs Centralized Comparison",
        fontsize=13, fontweight="bold",
    )
 
    # Optional centralized baseline
    if centr_cost is not None:
        axes[0].plot(centr_cost, color="black", linestyle="--", linewidth=1.5,
                     alpha=0.8, label="Centralized (Ideal)", zorder=2)
    if centr_grad is not None:
        axes[1].semilogy(centr_grad, color="black", linestyle="--",
                         linewidth=1.5, alpha=0.8,
                         label="Centralized (Ideal)", zorder=2)
 
    for idx, res in enumerate(results):
        lbl   = f"{res['graph_name'].capitalize()} | M={res['M']}"
        color = cmap(idx / max(len(results) - 1, 1))
 
        axes[0].plot(res["cost_history"],      label=lbl, color=color,
                     linewidth=1.8, zorder=3)
        axes[1].semilogy(res["grad_norm_history"], label=lbl, color=color,
                         linewidth=1.8, zorder=3)
        axes[2].semilogy(res["consensus_history"], label=lbl, color=color,
                         linewidth=1.8, zorder=3)
 
    specs = [
        (axes[0], r"Global Cost Function $\mathcal{L}$",              "Total Loss"),
        (axes[1], r"Gradient Norm $\|\nabla\mathcal{L}\|$ (Log Scale)", "Norm (Log Scale)"),
        (axes[2], r"Consensus Error $\|z - \bar{z}\|$ (Log Scale)",    "Consensus Error (Log Scale)"),
    ]
    for ax, title, ylabel in specs:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
 
    axes[-1].set_xlabel("Iteration $k$")
    plt.tight_layout(h_pad=2.0)
    return fig
 
 
def plot_task1_3_data_split(agents_data):
    """
    Visualises how the dataset is distributed among agents, with each agent
    assigned a distinct colour to highlight the spatial bias.
    Mirrors plot_data_split() from plot_utils.py.
 
    Parameters
    ----------
    agents_data : dict  {agent_index: {"X": …, "y": …}}
                  as produced by split_dataset_even_groups()
    """
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(
        "Task 1.3 – Dataset Distribution Among Agents",
        fontsize=13, fontweight="bold",
    )
 
    for i, data in agents_data.items():
        ax.scatter(data["X"][:, 0], data["X"][:, 1],
                   color=cmap(i), label=f"Agent {i + 1}",
                   alpha=0.7, edgecolors="k", s=25)
 
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
 
    # Legend outside the plot to avoid cluttering
    fig.subplots_adjust(right=0.78)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
 
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
#  Task 2.1
# ──────────────────────────────────────────────────────────────

#
# Distributed Autonomous Systems - Final Project
# Plotting utilities for Task 2.1 – Aggregative Tracking
# Simone Bernardi, Giorgio Soricetti
# Bologna, 02/06/2026
#

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
    return anim, fig
    
