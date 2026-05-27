"""
plots_task2.py  –  Plotting utilities for Task 2
=================================================
All matplotlib figures for Task 2.1 live here.

Functions
---------
plot_task2_1_metrics(scenario)        – cost, grad norm, σ error, consensus
plot_task2_1_trajectories(scenario)   – 2-D robot paths + targets + barycenter
plot_task2_1_animation(scenario)      – animated 2-D visualisation
plot_task2_1_comparison(scenarios)    – overlay metrics for multiple scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ── Palette ──────────────────────────────────────────────────────────────
_ROBOT_CMAP = plt.cm.tab10
_COLOR_BARYCENTER = "#e15759"
_COLOR_R0         = "#f28e2b"
_COLOR_TARGET      = "#76b7b2"


# ─────────────────────────────────────────────────────────────────────────
#  Helper: colour sequence for N robots
# ─────────────────────────────────────────────────────────────────────────

def _robot_colors(N):
    return [_ROBOT_CMAP(i / max(N - 1, 1)) for i in range(N)]


# ─────────────────────────────────────────────────────────────────────────
#  Plot 1 – Convergence metrics for one scenario
# ─────────────────────────────────────────────────────────────────────────

def plot_task2_1_metrics(scenario, title=None):
    """
    Four-panel convergence plot for one aggregative-tracking run.

    Panels: Total Cost | Gradient Norm | σ Estimation Error | Consensus Error

    Parameters
    ----------
    scenario : dict returned by run_task2_1()
    title    : optional override for the suptitle
    """
    m   = scenario["metrics"]
    lbl = scenario.get("label", "")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle(title or f"Task 2.1 – Convergence Metrics  |  {lbl}",
                 fontsize=12, fontweight="bold")

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


# ─────────────────────────────────────────────────────────────────────────
#  Plot 2 – 2-D robot trajectories
# ─────────────────────────────────────────────────────────────────────────

def plot_task2_1_trajectories(scenario, title=None, subsample=5):
    """
    2-D visualisation of robot paths, private targets, common target,
    barycenter trajectory, and final optimal positions.

    Parameters
    ----------
    scenario  : dict from run_task2_1()
    subsample : plot every k-th position along the trajectory (performance)
    """
    z_hist    = scenario["z_hist"]          # (K+1, N, 2)
    r_targets = scenario["r_targets"]       # (N, 2)
    r0        = scenario["r0"]              # (2,)
    z_init    = scenario["z_init"]          # (N, 2)
    z_opt     = scenario["z_opt"]           # (N, 2)
    sigma_opt = scenario["sigma_opt"]       # (2,)
    N         = scenario["N"]
    lbl       = scenario.get("label", "")

    colors = _robot_colors(N)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(title or f"Task 2.1 – Robot Trajectories  |  {lbl}",
                 fontsize=12, fontweight="bold")

    # ── Barycenter trajectory ────────────────────────────────
    sigma_hist = z_hist.mean(axis=1)        # (K+1, 2)
    ax.plot(sigma_hist[::subsample, 0], sigma_hist[::subsample, 1],
            color=_COLOR_BARYCENTER, linewidth=2.0, linestyle="--",
            zorder=3, label="Barycenter path")
    ax.scatter(*sigma_hist[0],  marker="^", s=90, color=_COLOR_BARYCENTER,
               zorder=5, edgecolors="k", linewidths=0.5)
    ax.scatter(*sigma_hist[-1], marker="*", s=180, color=_COLOR_BARYCENTER,
               zorder=5, edgecolors="k", linewidths=0.5)
    ax.scatter(*sigma_opt, marker="P", s=200, color=_COLOR_BARYCENTER,
               zorder=6, edgecolors="k", linewidths=0.8,
               label="σ* (optimal barycenter)")

    # ── Common target r₀ ────────────────────────────────────
    ax.scatter(*r0, marker="X", s=220, color=_COLOR_R0, zorder=6,
               edgecolors="k", linewidths=0.8, label="r₀ (common target)")

    # ── Per-robot trajectories ───────────────────────────────
    for i in range(N):
        c = colors[i]
        traj = z_hist[::subsample, i, :]
        ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=1.2,
                alpha=0.7, zorder=2)
        # start
        ax.scatter(*z_init[i], marker="o", s=70, color=c,
                   zorder=4, edgecolors="k", linewidths=0.5)
        # end
        ax.scatter(*z_hist[-1, i], marker="s", s=80, color=c,
                   zorder=4, edgecolors="k", linewidths=0.5)
        # optimal position
        ax.scatter(*z_opt[i], marker="*", s=150, color=c,
                   zorder=5, edgecolors="k", linewidths=0.5)
        # private target
        ax.scatter(*r_targets[i], marker="D", s=70,
                   color=c, alpha=0.5, zorder=3,
                   edgecolors=c, linewidths=1.2)
        # label
        ax.annotate(f"R{i}", z_init[i], fontsize=7, color=c,
                    xytext=(4, 4), textcoords="offset points")

    # ── Legend ──────────────────────────────────────────────
    legend_elems = [
        Line2D([0], [0], color=_COLOR_BARYCENTER, lw=2, ls="--",
               label="Barycenter path"),
        plt.scatter([], [], marker="P", s=80, c=_COLOR_BARYCENTER,
                    label="σ* (optimal bary.)"),
        plt.scatter([], [], marker="X", s=80, c=_COLOR_R0,
                    label="r₀ (common target)"),
        Line2D([0], [0], color="gray", lw=1.2, label="Robot path"),
        plt.scatter([], [], marker="o", s=50, c="gray", label="Start"),
        plt.scatter([], [], marker="s", s=50, c="gray", label="Final"),
        plt.scatter([], [], marker="*", s=100, c="gray", label="Optimal z*"),
        plt.scatter([], [], marker="D", s=50, c="gray", alpha=0.5,
                    label="Private target rᵢ"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="best",
              framealpha=0.85, ncol=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
#  Plot 3 – Animated 2-D visualisation
# ─────────────────────────────────────────────────────────────────────────

def plot_task2_1_animation(scenario, step=5, interval=60, title=None):
    """
    Animated visualisation of robots moving towards their targets.

    Parameters
    ----------
    scenario : dict from run_task2_1()
    step     : frame stride (every `step` iterations = 1 frame)
    interval : ms between frames

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation  (call plt.show() or anim.save())
    fig  : Figure
    """
    z_hist    = scenario["z_hist"]          # (K+1, N, 2)
    r_targets = scenario["r_targets"]
    r0        = scenario["r0"]
    z_opt     = scenario["z_opt"]
    N         = scenario["N"]
    lbl       = scenario.get("label", "")

    frames    = list(range(0, z_hist.shape[0], step))
    colors    = _robot_colors(N)

    # Compute axis limits from all positions + targets
    all_pos = np.vstack([z_hist.reshape(-1, 2), r_targets, r0[None, :],
                         z_opt])
    margin  = 1.5
    xlim    = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim    = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle(title or f"Task 2.1 – Animation  |  {lbl}",
                 fontsize=11, fontweight="bold")

    # Static elements
    for i in range(N):
        ax.scatter(*r_targets[i], marker="D", s=80, color=colors[i],
                   alpha=0.4, edgecolors=colors[i], linewidths=1.2, zorder=2)
        ax.scatter(*z_opt[i], marker="*", s=120, color=colors[i],
                   alpha=0.5, edgecolors="k", linewidths=0.4, zorder=2)
    ax.scatter(*r0, marker="X", s=200, color=_COLOR_R0,
               edgecolors="k", linewidths=0.8, zorder=3, label="r₀")

    # Dynamic elements – one scatter per robot + barycenter
    robot_dots   = [ax.plot([], [], "o", color=colors[i], ms=9,
                             markeredgecolor="k", markeredgewidth=0.5,
                             zorder=5)[0]
                    for i in range(N)]
    robot_trails = [ax.plot([], [], "-", color=colors[i], lw=1.0,
                             alpha=0.5, zorder=3)[0]
                    for i in range(N)]
    bary_dot,  = ax.plot([], [], "^", color=_COLOR_BARYCENTER, ms=11,
                          markeredgecolor="k", markeredgewidth=0.5, zorder=6)
    bary_trail, = ax.plot([], [], "--", color=_COLOR_BARYCENTER, lw=1.8,
                           alpha=0.7, zorder=4)
    iter_text   = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                          fontsize=9, va="top")

    # Legend
    legend_elems = [
        Line2D([0], [0], marker="o", color="gray", ms=8, lw=0,
               label="Robot"),
        Line2D([0], [0], marker="^", color=_COLOR_BARYCENTER, ms=9, lw=0,
               label="Barycenter"),
        Line2D([0], [0], marker="X", color=_COLOR_R0, ms=9, lw=0,
               label="r₀"),
        Line2D([0], [0], marker="D", color="gray", ms=7, lw=0, alpha=0.5,
               label="Private target"),
        Line2D([0], [0], marker="*", color="gray", ms=10, lw=0, alpha=0.5,
               label="Optimal z*"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right",
              framealpha=0.85)

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
        trail_k = frames[max(0, f_idx - 20): f_idx + 1]

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

    anim = animation.FuncAnimation(fig, _update, init_func=_init,
                                   frames=len(frames), interval=interval,
                                   blit=True, repeat=True)
    return anim, fig


# ─────────────────────────────────────────────────────────────────────────
#  Plot 4 – Overlaid metric comparison across multiple scenarios
# ─────────────────────────────────────────────────────────────────────────

def plot_task2_1_comparison(scenarios, title=None):
    """
    Overlay convergence metrics for multiple scenarios in a single figure.

    Parameters
    ----------
    scenarios : list of dicts from run_task2_1()

    Returns
    -------
    fig : Figure
    """
    SCENARIO_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    SCENARIO_LS     = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    metric_keys  = ["cost",       "grad_norm",    "sigma_error",     "consensus"]
    metric_titles = ["Total Cost", "Gradient Norm", "σ Estimation Error", "Consensus Error"]
    metric_log    = [False, True, True, True]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle(title or "Task 2.1 – Scenario Comparison",
                 fontsize=12, fontweight="bold")

    handles = []
    for s_idx, sc in enumerate(scenarios):
        color = SCENARIO_COLORS[s_idx % len(SCENARIO_COLORS)]
        ls    = SCENARIO_LS[s_idx % len(SCENARIO_LS)]
        label = sc.get("label", f"Scenario {s_idx}")

        for ax, key, t, use_log in zip(axes, metric_keys, metric_titles, metric_log):
            data = sc["metrics"][key]
            if use_log:
                line, = ax.semilogy(data, color=color, linestyle=ls,
                                    linewidth=1.8, label=label)
            else:
                line, = ax.plot(data, color=color, linestyle=ls,
                                linewidth=1.8, label=label)
            ax.set_title(t, fontsize=10)
            ax.set_xlabel("Iterations")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)

            if ax is axes[0]:
                handles.append(line)

    fig.legend(handles=handles, loc="lower center", ncol=len(scenarios),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────
#  Plot 5 – Animated 2-D visualisation for Task 2.3 (Nominal vs Safe)
# ─────────────────────────────────────────────────────────────────────────

def plot_task2_3_animation(nominal, safe, step=5, interval=60, title=None):
    """
    Animated side-by-side visualisation of robots navigating obstacles.
    """
    z_nom  = nominal["z"]          
    z_safe = safe["z"]             
    N      = z_nom.shape[1]
    
    frames = list(range(0, z_nom.shape[0], step))
    colors = _robot_colors(N)

    # Compute axis limits dynamically from initial and target positions
    all_pos = np.vstack([nominal["z"][0], nominal["targets"], nominal["obstacles"]])
    margin  = 2.0
    xlim    = (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ylim    = (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title or f"Task 2.3 – CBF-QP Safety Animation", fontsize=12, fontweight="bold")

    # Static elements setup for both subplots
    for ax, data, title_ax in zip(axes, [nominal, safe], ["Nominal Tracking", "CBF Safe Tracking"]):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(title_ax)

        # Draw Obstacles
        for o_idx, center in enumerate(data['obstacles']):
            circle = plt.Circle(center, data['d_safe'], facecolor='tab:orange', edgecolor='tab:red', alpha=0.25, linewidth=2)
            ax.add_patch(circle)
            ax.text(center[0] + 0.2, center[1] + 0.2, f'O{o_idx + 1}', color='tab:red', fontsize=10)

        # Draw Targets
        for i in range(N):
            ax.scatter(*data['targets'][i], marker="x", s=100, color=colors[i], zorder=2)

    # Dynamic elements – one scatter & trail per robot per subplot
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
        k = frames[f_idx]
        
        for i in range(N):
            # Update Nominal
            dots_nom[i].set_data([z_nom[k, i, 0]], [z_nom[k, i, 1]])
            trail_nom_x = [z_nom[frames[t], i, 0] for t in range(max(0, f_idx-20), f_idx+1)]
            trail_nom_y = [z_nom[frames[t], i, 1] for t in range(max(0, f_idx-20), f_idx+1)]
            trails_nom[i].set_data(trail_nom_x, trail_nom_y)

            # Update Safe
            dots_safe[i].set_data([z_safe[k, i, 0]], [z_safe[k, i, 1]])
            trail_safe_x = [z_safe[frames[t], i, 0] for t in range(max(0, f_idx-20), f_idx+1)]
            trail_safe_y = [z_safe[frames[t], i, 1] for t in range(max(0, f_idx-20), f_idx+1)]
            trails_safe[i].set_data(trail_safe_x, trail_safe_y)

        iter_text.set_text(f"k = {k}")
        return dots_nom + trails_nom + dots_safe + trails_safe + [iter_text]

    anim = animation.FuncAnimation(fig, _update, init_func=_init, frames=len(frames), interval=interval, blit=False, repeat=True)
    return anim, fig