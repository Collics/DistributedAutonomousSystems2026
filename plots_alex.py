import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────
#  Task 1.1
# ──────────────────────────────────────────────────────────────
 
def plot_task1_1(cost, gradient, consensus, title="Task 1.1 – Distributed Gradient Tracking"):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
 
    axes[0].plot(cost[:-1], linewidth=2)
    axes[0].set_title(r"Evolution of the Global Cost Function $\ell(\bar{z})$")
    axes[0].set_ylabel("Cost Function")
    axes[0].grid(True, alpha=0.4)
 
    axes[1].semilogy(np.abs(gradient[:-1]), linewidth=2)
    axes[1].set_title(r"Evolution of the Global Gradient Norm $\|\nabla\ell(\bar{z})\|$ (Log Scale)")
    axes[1].set_ylabel("Norm of Gradient (Log Scale)")
    axes[1].grid(True, which="both", alpha=0.4)
 
    axes[2].semilogy(consensus[:-1], linewidth=2)
    axes[2].set_title(r"Evolution of the Consensus Error $\|z - \bar{z}\|$")
    axes[2].set_ylabel("Consensus Error (Log Scale)")
    axes[2].set_xlabel("Iteration $k$")
    axes[2].grid(True, which="both", alpha=0.4)
 
    plt.tight_layout(h_pad=2.0)
    return fig
 
def plot_task1_1_comparison(results):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    fig.suptitle("Task 1.1 – Topology Comparison", fontsize=13, fontweight="bold")
 
    for g_type, m in results.items():
        lbl = f"Topology: {g_type.capitalize()}"
        axes[0].plot(m["cost"][:-1],              label=lbl, linewidth=2)
        axes[1].semilogy(np.abs(m["gradient"][:-1]), label=lbl, linewidth=2)
        axes[2].semilogy(m["consensus"][:-1],     label=lbl, linewidth=2)
 
    specs = [
        (axes[0], r"Evolution of the Global Cost Function $\ell(\bar{z})$", "Cost Function", False),
        (axes[1], r"Evolution of the Global Gradient Norm $\|\nabla\ell(\bar{z})\|$ (Log Scale)", "Norm of Gradient (Log Scale)", True),
        (axes[2], r"Evolution of the Consensus Error $\|z - \bar{z}\|$ (Log Scale)", "Consensus Error (Log Scale)", True),
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
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f"Topology: {graph_type.capitalize()}", fontsize=14, fontweight="bold")
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue", edge_color="gray", node_size=600, font_weight="bold")
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────
#  Task 1.2
# ──────────────────────────────────────────────────────────────

def plot_task1_2(cost_hist, grad_norm_hist, map_name=""):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Task 1.2 – Centralised Logistic Regression ({map_name})", fontsize=13, fontweight="bold")
 
    axes[0].plot(cost_hist, linewidth=2)
    axes[0].set_title(r"Evolution of the Loss $\mathcal{L}(w, b)$")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.4)
 
    axes[1].semilogy(grad_norm_hist, linewidth=2)
    axes[1].set_title(r"Evolution of the Gradient Norm $\|\nabla\mathcal{L}\|$ (Log Scale)")
    axes[1].set_ylabel(r"$\|\nabla\mathcal{L}\|$ (Log Scale)")
    axes[1].set_xlabel("Iteration $k$")
    axes[1].grid(True, which="both", alpha=0.4)
 
    plt.tight_layout(h_pad=2.0)
    return fig
 
def plot_task1_2_dataset(X, labels, phi_fn=None, wb=None, map_name="", data_range=(-3, 3)):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Task 1.2 – Dataset & Decision Boundary ({map_name})", fontsize=13, fontweight="bold")
 
    ax.scatter(X[labels ==  1, 0], X[labels ==  1, 1], color="blue", label="Class +1", alpha=0.5, s=15)
    ax.scatter(X[labels == -1, 0], X[labels == -1, 1], color="red",  label="Class −1", alpha=0.5, s=15)
 
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
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
    fig.suptitle(title or "Task 1.3 – Distributed Logistic Regression", fontsize=13, fontweight="bold")
 
    axes[0].plot(cost_history, linewidth=2)
    axes[0].set_title(r"Evolution of the Global Cost Function $\mathcal{L}$")
    axes[0].set_ylabel("Total Loss")
    axes[0].grid(True, alpha=0.4)
 
    axes[1].semilogy(grad_norm_history, linewidth=2)
    axes[1].set_title(r"Evolution of the Gradient Norm $\|\nabla\mathcal{L}\|$ (Log Scale)")
    axes[1].set_ylabel("Norm (Log Scale)")
    axes[1].grid(True, which="both", alpha=0.4)
 
    axes[2].semilogy(consensus_history, linewidth=2)
    axes[2].set_title(r"Evolution of the Consensus Error $\|z - \bar{z}\|$ (Log Scale)")
    axes[2].set_ylabel("Consensus Error (Log Scale)")
    axes[2].set_xlabel("Iteration $k$")
    axes[2].grid(True, which="both", alpha=0.4)
 
    plt.tight_layout(h_pad=2.0)
    return fig
 
def plot_task1_3_comparison(results, centr_cost=None, centr_grad=None):
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    fig.suptitle("Task 1.3 – Distributed vs Centralized Comparison", fontsize=13, fontweight="bold")
 
    if centr_cost is not None:
        axes[0].plot(centr_cost, color="black", linestyle="--", linewidth=1.5, alpha=0.8, label="Centralized (Ideal)", zorder=2)
    if centr_grad is not None:
        axes[1].semilogy(centr_grad, color="black", linestyle="--", linewidth=1.5, alpha=0.8, label="Centralized (Ideal)", zorder=2)
 
    for idx, res in enumerate(results):
        lbl   = f"{res['graph_name'].capitalize()} | M={res['M']}"
        color = cmap(idx / max(len(results) - 1, 1))
        axes[0].plot(res["cost_history"],      label=lbl, color=color, linewidth=1.8, zorder=3)
        axes[1].semilogy(res["grad_norm_history"], label=lbl, color=color, linewidth=1.8, zorder=3)
        axes[2].semilogy(res["consensus_history"], label=lbl, color=color, linewidth=1.8, zorder=3)
 
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
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Task 1.3 – Dataset Distribution Among Agents", fontsize=13, fontweight="bold")
 
    for i, data in agents_data.items():
        ax.scatter(data["X"][:, 0], data["X"][:, 1], color=cmap(i), label=f"Agent {i + 1}", alpha=0.7, edgecolors="k", s=25)
 
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.subplots_adjust(right=0.78)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
#  Task 2 Shared Variables
# ──────────────────────────────────────────────────────────────

_ROBOT_CMAP = plt.cm.tab10
_COLOR_BARYCENTER = "#e15759"
_COLOR_TARGET = "#76b7b2"

def _robot_colors(N):
    return [_ROBOT_CMAP(i / max(N - 1, 1)) for i in range(N)]


# ──────────────────────────────────────────────────────────────
#  Task 2.1
# ──────────────────────────────────────────────────────────────

def plot_task2_1_metrics(scenario, title=None):
    m   = scenario["metrics"]
    lbl = scenario.get("label", "")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle(title or f"Task 2.1 – Convergence Metrics  |  {lbl}", fontsize=12, fontweight="bold")

    specs = [
        (m["cost"],        "Total Cost",          False, "#1f77b4"),
        (m["grad_norm"],   "Gradient Norm",        True,  "#d62728"),
        (m["sigma_error"], "σ Estimation Error",   True,  "#2ca02c"),
        (m["consensus"],   "Consensus Error (s)",  True,  "#9467bd"),
    ]

    for ax, (data, title_ax, use_log, color) in zip(axes, specs):
        if use_log:
            ax.semilogy(data, color=color, linewidth=1.6)
        else:
            ax.plot(data, color=color, linewidth=1.6)
        ax.set_title(title_ax, fontsize=10)
        ax.set_xlabel("Iterations")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig

def plot_task2_1_trajectories(scenario, title=None, subsample=5):
    z_hist    = scenario["z_hist"]          
    r_targets = scenario["r_targets"]       
    z_init    = scenario["z_init"]          
    z_opt     = scenario["z_opt"]           
    sigma_opt = scenario["sigma_opt"]       
    N         = scenario["N"]
    lbl       = scenario.get("label", "")

    colors = _robot_colors(N)
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(title or f"Task 2.1 – Robot Trajectories  |  {lbl}", fontsize=12, fontweight="bold")

    sigma_hist = z_hist.mean(axis=1)        
    ax.plot(sigma_hist[::subsample, 0], sigma_hist[::subsample, 1], color=_COLOR_BARYCENTER, linewidth=2.0, linestyle="--", zorder=3, label="Barycenter path")
    ax.scatter(*sigma_hist[0],  marker="^", s=90, color=_COLOR_BARYCENTER, zorder=5, edgecolors="k", linewidths=0.5)
    ax.scatter(*sigma_hist[-1], marker="*", s=180, color=_COLOR_BARYCENTER, zorder=5, edgecolors="k", linewidths=0.5)
    ax.scatter(*sigma_opt, marker="P", s=200, color=_COLOR_BARYCENTER, zorder=6, edgecolors="k", linewidths=0.8, label="σ* (optimal barycenter)")

    for i in range(N):
        c = colors[i]
        traj = z_hist[::subsample, i, :]
        ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=1.2, alpha=0.7, zorder=2)
        ax.scatter(*z_init[i], marker="o", s=70, color=c, zorder=4, edgecolors="k", linewidths=0.5)
        ax.scatter(*z_hist[-1, i], marker="s", s=80, color=c, zorder=4, edgecolors="k", linewidths=0.5)
        ax.scatter(*z_opt[i], marker="*", s=150, color=c, zorder=5, edgecolors="k", linewidths=0.5)
        ax.scatter(*r_targets[i], marker="D", s=70, color=c, alpha=0.5, zorder=3, edgecolors=c, linewidths=1.2)
        ax.annotate(f"R{i}", z_init[i], fontsize=7, color=c, xytext=(4, 4), textcoords="offset points")

    legend_elems = [
        Line2D([0], [0], color=_COLOR_BARYCENTER, lw=2, ls="--", label="Barycenter path"),
        plt.scatter([], [], marker="P", s=80, c=_COLOR_BARYCENTER, label="σ* (optimal bary.)"),
        Line2D([0], [0], color="gray", lw=1.2, label="Robot path"),
        plt.scatter([], [], marker="o", s=50, c="gray", label="Start"),
        plt.scatter([], [], marker="s", s=50, c="gray", label="Final"),
        plt.scatter([], [], marker="*", s=100, c="gray", label="Optimal z*"),
        plt.scatter([], [], marker="D", s=50, c="gray", alpha=0.5, label="Private target rᵢ"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="best", framealpha=0.85, ncol=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_task2_1_animation(scenario, step=5, interval=60, title=None):
    z_hist    = scenario["z_hist"]          
    r_targets = scenario["r_targets"]
    z_opt     = scenario["z_opt"]
    N         = scenario["N"]
    lbl       = scenario.get("label", "")

    frames    = list(range(0, z_hist.shape[0], step))
    colors    = _robot_colors(N)

    all_pos = np.vstack([z_hist.reshape(-1, 2), r_targets, z_opt])
    margin  = 1.5
    xlim    = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim    = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle(title or f"Task 2.1 – Animation  |  {lbl}", fontsize=11, fontweight="bold")

    for i in range(N):
        ax.scatter(*r_targets[i], marker="D", s=80, color=colors[i], alpha=0.4, edgecolors=colors[i], linewidths=1.2, zorder=2)
        ax.scatter(*z_opt[i], marker="*", s=120, color=colors[i], alpha=0.5, edgecolors="k", linewidths=0.4, zorder=2)

    robot_dots   = [ax.plot([], [], "o", color=colors[i], ms=9, markeredgecolor="k", markeredgewidth=0.5, zorder=5)[0] for i in range(N)]
    robot_trails = [ax.plot([], [], "-", color=colors[i], lw=1.0, alpha=0.5, zorder=3)[0] for i in range(N)]
    bary_dot,  = ax.plot([], [], "^", color=_COLOR_BARYCENTER, ms=11, markeredgecolor="k", markeredgewidth=0.5, zorder=6)
    bary_trail, = ax.plot([], [], "--", color=_COLOR_BARYCENTER, lw=1.8, alpha=0.7, zorder=4)
    iter_text   = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=9, va="top")

    legend_elems = [
        Line2D([0], [0], marker="o", color="gray", ms=8, lw=0, label="Robot"),
        Line2D([0], [0], marker="^", color=_COLOR_BARYCENTER, ms=9, lw=0, label="Barycenter"),
        Line2D([0], [0], marker="D", color="gray", ms=7, lw=0, alpha=0.5, label="Private target"),
        Line2D([0], [0], marker="*", color="gray", ms=10, lw=0, alpha=0.5, label="Optimal z*"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right", framealpha=0.85)

    def _init():
        for d in robot_dots + robot_trails:
            d.set_data([], [])
        bary_dot.set_data([], [])
        bary_trail.set_data([], [])
        iter_text.set_text("")
        return robot_dots + robot_trails + [bary_dot, bary_trail, iter_text]

    def _update(f_idx):
        k = frames[f_idx]
        sigma = z_hist[k].mean(axis=0)

        for i in range(N):
            robot_dots[i].set_data([z_hist[k, i, 0]], [z_hist[k, i, 1]])
            trail_x = [z_hist[frames[t], i, 0] for t in range(max(0, f_idx-20), f_idx+1)]
            trail_y = [z_hist[frames[t], i, 1] for t in range(max(0, f_idx-20), f_idx+1)]
            robot_trails[i].set_data(trail_x, trail_y)

        bary_dot.set_data([sigma[0]], [sigma[1]])
        bary_x = [z_hist[frames[t]].mean(axis=0)[0] for t in range(max(0, f_idx-20), f_idx+1)]
        bary_y = [z_hist[frames[t]].mean(axis=0)[1] for t in range(max(0, f_idx-20), f_idx+1)]
        bary_trail.set_data(bary_x, bary_y)
        iter_text.set_text(f"k = {k}")
        return robot_dots + robot_trails + [bary_dot, bary_trail, iter_text]

    anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=len(frames), interval=interval, blit=True, repeat=False)
    return anim, fig

def plot_task2_1_comparison(scenarios, title=None):
    SCENARIO_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    SCENARIO_LS     = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    metric_keys  = ["cost",       "grad_norm",    "sigma_error",     "consensus"]
    metric_titles = ["Total Cost", "Gradient Norm", "σ Estimation Error", "Consensus Error"]
    metric_log    = [False, True, True, True]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle(title or "Task 2.1 – Scenario Comparison", fontsize=12, fontweight="bold")

    handles = []
    for s_idx, sc in enumerate(scenarios):
        color = SCENARIO_COLORS[s_idx % len(SCENARIO_COLORS)]
        ls    = SCENARIO_LS[s_idx % len(SCENARIO_LS)]
        label = sc.get("label", f"Scenario {s_idx}")

        for ax, key, t, use_log in zip(axes, metric_keys, metric_titles, metric_log):
            data = sc["metrics"][key]
            if use_log:
                line, = ax.semilogy(data, color=color, linestyle=ls, linewidth=1.8, label=label)
            else:
                line, = ax.plot(data, color=color, linestyle=ls, linewidth=1.8, label=label)
            ax.set_title(t, fontsize=10)
            ax.set_xlabel("Iterations")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)

            if ax is axes[0]:
                handles.append(line)

    fig.legend(handles=handles, loc="lower center", ncol=len(scenarios), fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ──────────────────────────────────────────────────────────────
#  Task 2.3
# ──────────────────────────────────────────────────────────────

def _draw_obstacles(ax, obstacles, d_safe):
    for idx, center in enumerate(obstacles):
        circle = plt.Circle(center, d_safe, facecolor='tab:orange', edgecolor='tab:red', alpha=0.25, linewidth=2)
        ax.add_patch(circle)
        ax.text(center[0] + 0.2, center[1] + 0.2, f'O{idx + 1}', color='tab:red', fontsize=10)

def plot_task2_3_trajectories(nominal, safe, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title or "Task 2.3 – Trajectory Comparison", fontsize=13, fontweight="bold")

    for idx, (data, title_ax) in enumerate([
        (nominal, f"Nominal Tracking - {nominal['graph_type']}"),
        (safe, f"CBF Safe Tracking - {safe['graph_type']}")
    ]):
        ax = axes[idx]
        _draw_obstacles(ax, data['obstacles'], data['d_safe'])
        for i in range(data['z'].shape[1]):
            ax.plot(data['z'][:, i, 0], data['z'][:, i, 1], alpha=0.7, label=f'Robot {i + 1}')
            ax.scatter(data['z'][0, i, 0], data['z'][0, i, 1], marker='o', color='blue', alpha=0.5)
            ax.scatter(data['targets'][i, 0], data['targets'][i, 1], marker='x', s=100, color='red')
        ax.set_title(title_ax)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axis('equal')
    fig.tight_layout()
    return fig

def plot_task2_3_metrics(nominal, safe, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title or "Task 2.3 – Convergence Metrics", fontsize=13, fontweight="bold")

    axes[0].plot(nominal['cost_history'], label='Nominal', linewidth=2, color='tab:blue')
    axes[0].plot(safe['cost_history'], label='Safe (CBF)', linewidth=2, linestyle='--', color='tab:red')
    axes[0].set_title('Global Cost Evolution')
    axes[0].set_xlabel('Iteration k')
    axes[0].set_ylabel(r'$J(z,\sigma)$')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    axes[1].semilogy(nominal['grad_norm_history'], label='Nominal', linewidth=2, color='tab:orange')
    axes[1].semilogy(safe['grad_norm_history'], label='Safe (CBF)', linewidth=2, linestyle='--', color='tab:red')
    axes[1].set_title('Gradient Norm Evolution')
    axes[1].set_xlabel('Iteration k')
    axes[1].set_ylabel(r'$\| \nabla J(z,\sigma) \|$')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    fig.tight_layout()
    return fig

def plot_task2_3_safety(nominal, safe, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title or "Task 2.3 – Safety Metrics", fontsize=13, fontweight="bold")

    axes[0].plot(nominal['min_clearance'], label='Nominal', linewidth=2)
    axes[0].plot(safe['min_clearance'], label='CBF-QP safe', linewidth=2)
    axes[0].axhline(0.0, color='k', linestyle='--', linewidth=1)
    axes[0].set_title('Minimum clearance to obstacle boundary')
    axes[0].set_xlabel('Iteration k')
    axes[0].set_ylabel(r'$\|z_i - p_o\| - d_{safe}$')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    control_delta = np.linalg.norm(safe['u_app'] - nominal['u_nom'], axis=2)
    axes[1].plot(np.max(control_delta, axis=1), color='tab:purple', linewidth=2)
    axes[1].set_title('Maximum safety correction per iteration')
    axes[1].set_xlabel('Iteration k')
    axes[1].set_ylabel(r'$\max(\|u_i^{safe} - u_i^{nom}\|)$')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()
    return fig

def plot_task2_3_animation(nominal, safe, step=5, interval=60, title=None):
    z_nom  = nominal["z"]          
    z_safe = safe["z"]             
    N      = z_nom.shape[1]
    
    frames = list(range(0, z_nom.shape[0], step))
    colors = _robot_colors(N)

    all_pos = np.vstack([nominal["z"][0], nominal["targets"], nominal["obstacles"]])
    margin  = 2.0
    xlim    = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim    = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title or f"Task 2.3 – CBF-QP Safety Animation", fontsize=12, fontweight="bold")

    for ax, data, title_ax in zip(axes, [nominal, safe], ["Nominal Tracking", "CBF Safe Tracking"]):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(title_ax)

        for o_idx, center in enumerate(data['obstacles']):
            circle = plt.Circle(center, data['d_safe'], facecolor='tab:orange', edgecolor='tab:red', alpha=0.25, linewidth=2)
            ax.add_patch(circle)
            ax.text(center[0] + 0.2, center[1] + 0.2, f'O{o_idx + 1}', color='tab:red', fontsize=10)

        for i in range(N):
            ax.scatter(*data['targets'][i], marker="x", s=100, color=colors[i], zorder=2)

    dots_nom = [axes[0].plot([], [], "o", color=colors[i], ms=9, markeredgecolor="k", markeredgewidth=0.5, zorder=5)[0] for i in range(N)]
    trails_nom = [axes[0].plot([], [], "-", color=colors[i], lw=1.0, alpha=0.5, zorder=3)[0] for i in range(N)]

    dots_safe = [axes[1].plot([], [], "o", color=colors[i], ms=9, markeredgecolor="k", markeredgewidth=0.5, zorder=5)[0] for i in range(N)]
    trails_safe = [axes[1].plot([], [], "-", color=colors[i], lw=1.0, alpha=0.5, zorder=3)[0] for i in range(N)]

    iter_text = axes[0].text(0.02, 0.97, "", transform=axes[0].transAxes, fontsize=9, va="top")

    def _init():
        for d in dots_nom + trails_nom + dots_safe + trails_safe:
            d.set_data([], [])
        iter_text.set_text("")
        return dots_nom + trails_nom + dots_safe + trails_safe + [iter_text]

    def _update(f_idx):
        k = frames