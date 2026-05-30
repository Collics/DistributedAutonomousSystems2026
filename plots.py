import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import Parameters as par
 
#  Task 1.1
def plot_task_1_1_summary_results(all_costs, all_grads, all_consensus):
    topologies = list(all_costs.keys())
    num_topologies = len(topologies)
    
    # Grid: num_topologies rows (one per topology), 3 columns (Cost, Gradient Norm, Consensus Error)
    fig, axes = plt.subplots(nrows=num_topologies, ncols=3, figsize=(15, 3 * num_topologies), sharex=True)
    
    # Handle single topology case where matplotlib returns a 1D array of axes
    if num_topologies == 1:
        axes = np.expand_dims(axes, axis=0)
        
    cmap = plt.get_cmap('Dark2')
    
    for row_idx, gt in enumerate(topologies):
        color = cmap(row_idx)
        
        # 1. Cost function plot (Column 0)
        ax_cost = axes[row_idx, 0]
        ax_cost.plot(all_costs[gt], linewidth=2.0, color=color)
        ax_cost.set_ylabel(f"{gt.capitalize()}\nCost Function")
        if row_idx == 0:
            ax_cost.set_title(r"Global Cost $l(\bar{z})$", fontweight='bold')
        
        # 2. Gradient norm plot (Column 1)
        ax_grad = axes[row_idx, 1]
        ax_grad.semilogy(all_grads[gt], linewidth=2.0, color=color)
        ax_grad.set_ylabel("Gradient Norm (Log)")
        if row_idx == 0:
            ax_grad.set_title(r"Gradient Norm $\|\nabla l(\bar{z})\|$", fontweight='bold')
            
        # 3. Consensus error plot (Column 2)
        ax_consensus = axes[row_idx, 2]
        ax_consensus.plot(all_consensus[gt], linewidth=2.0, color=color)
        ax_consensus.set_ylabel("Consensus Error")
        if row_idx == 0:
            ax_consensus.set_title(r"Consensus Error $\|z - \bar{z}\|$", fontweight='bold')
            
        # Set x-label for the last row
        if row_idx == num_topologies - 1:
            ax_cost.set_xlabel("Iterations k")
            ax_grad.set_xlabel("Iterations k")
            ax_consensus.set_xlabel("Iterations k")
            
    plt.tight_layout()
    plt.show()

def plot_task_1_1_overlying_summary_results(all_costs, all_grads, all_consensus):
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

def plot_task_1_1_single_topology_results(cost_hist, grad_hist, consensus_hist, topology_name):
    """
    Plots the cost function, gradient norm, and consensus error for a single topology.
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"Convergence Metrics for Topology: {topology_name.upper()}", fontsize=14, fontweight='bold')
    
    # 1. Cost Function
    axes[0].plot(cost_hist, color='#20B2AA', linewidth=2.5)
    axes[0].set_title(r"Global Cost Function $l(\bar{z})$", fontweight='bold')
    axes[0].set_ylabel("Cost")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Gradient Norm
    axes[1].semilogy(grad_hist, color='#C71585', linewidth=2.5)
    axes[1].set_title(r"Global Gradient Norm $\|\nabla l(\bar{z})\|$ (Log Scale)", fontweight='bold')
    axes[1].set_ylabel("Norm (Log)")
    axes[1].grid(True, which="both", alpha=0.3)
    
    # 3. Consensus Error
    axes[2].plot(consensus_hist, color='#D9A441', linewidth=2.5)
    axes[2].set_title(r"Consensus Error $\|z - \bar{z}\|$", fontweight='bold')
    axes[2].set_ylabel("Consensus Error")
    axes[2].set_xlabel("Iterations k")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig

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
                   color='#8A2BE2', marker='^', label='+1', alpha=0.6, s=25, edgecolors='none')
        ax.scatter(X[conf["labels"] == -1, 0], X[conf["labels"] == -1, 1], 
                   color='#FF8C00', marker='v', label='-1', alpha=0.6, s=25, edgecolors='none')
        
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


def plot_task_1_2_boundary_comparison(X, labels_cubic, labels_super, phi_c, phi_s, 
                                      w_true_c, b_true_c, w_learned_c, b_learned_c, 
                                      w_true_s, b_true_s, w_learned_s, b_learned_s, 
                                      title_prefix=""):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle(f"{title_prefix}", fontsize=14, fontweight="bold")
    
    configs = [
        {
            "name": "Parabola",
            "labels": labels_cubic,
            "phi": phi_c,
            "w_true": w_true_c,
            "b_true": b_true_c,
            "w_learned": w_learned_c,
            "b_learned": b_learned_c,
            "ax": axes[0]
        },
        {
            "name": "Hyperbola",
            "labels": labels_super,
            "phi": phi_s,
            "w_true": w_true_s,
            "b_true": b_true_s,
            "w_learned": w_learned_s,
            "b_learned": b_learned_s,
            "ax": axes[1]
        }
    ]
    
    x_range = np.linspace(par.TASK_1_2_RANGE[0], par.TASK_1_2_RANGE[1], 100)
    y_range = np.linspace(par.TASK_1_2_RANGE[0], par.TASK_1_2_RANGE[1], 100)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    pts = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])

    for conf in configs:
        ax = conf["ax"]
        labels = conf["labels"]
        phi = conf["phi"]
        
        # Scatter points (matching Task 1.2 style)
        ax.scatter(X[labels == 1, 0], X[labels == 1, 1], 
                   color='#8A2BE2', marker='^', label='+1', alpha=0.6, s=25, edgecolors='none')
        ax.scatter(X[labels == -1, 0], X[labels == -1, 1], 
                   color='#FF8C00', marker='v', label='-1', alpha=0.6, s=25, edgecolors='none')
        
        Phi_grid = phi(pts)
        
        # 1. Plot True Boundary (slate gray dash-dotted line)
        if conf["w_true"] is not None and conf["b_true"] is not None:
            Z_true = (Phi_grid @ conf["w_true"] + conf["b_true"]).reshape(X_mesh.shape)
            ax.contour(X_mesh, Y_mesh, Z_true, levels=[0], colors='#2F4F4F', linestyles='-.', linewidths=2.5)
            ax.plot([], [], color='#2F4F4F', linestyle='-.', linewidth=2.5, label='True Boundary (Real)')
            
        # 2. Plot Learned Boundary (royal blue solid line)
        if conf["w_learned"] is not None and conf["b_learned"] is not None:
            Z_learned = (Phi_grid @ conf["w_learned"] + conf["b_learned"]).reshape(X_mesh.shape)
            ax.contour(X_mesh, Y_mesh, Z_learned, levels=[0], colors='#4169E1', linestyles='-', linewidths=2.5)
            ax.plot([], [], color='#4169E1', linestyle='-', linewidth=2.5, label='Learned Boundary')

        ax.set_title(f"{conf['name']} Feature Mapping", fontweight='bold')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect('equal')
        ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.show()
    return fig

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
        ax.scatter(data[:, 0], data[:, 1], color=cmap(i), marker='s', label=f"Node {i + 1}", alpha=0.75, edgecolors="white", s=45)
 
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
 
    fig.subplots_adjust(right=0.78)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10, framealpha=0.9)
 
    plt.tight_layout()
    plt.show() 
    return fig


def plot_task1_3_dataset_boundary(agents_data, phi_fn=None, wb=None, map_name="", data_range=(-2, 2)):
    # Side-by-side subplots: Left = Global Classification (Task 1.2 style), Right = Agent Data Split & Boundaries
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle(f"Task 1.3 – Distributed Classification ({map_name})", fontsize=14, fontweight="bold")
    
    # Aggregate points from all agents for the left plot
    X_all = []
    labels_all = []
    for agent in agents_data:
        X_all.append(agent["X"])
        labels_all.append(agent["labels"])
    
    X = np.vstack(X_all)
    labels = np.concatenate(labels_all)
    
    # ── LEFT PLOT: Global classification (Task 1.2 style) ───────────────────────────
    ax_left = axes[0]
    ax_left.scatter(X[labels == 1, 0], X[labels == 1, 1], 
                    color='#8A2BE2', marker='^', label='+1', alpha=0.6, s=25, edgecolors='none')
    ax_left.scatter(X[labels == -1, 0], X[labels == -1, 1], 
                    color='#FF8C00', marker='v', label='-1', alpha=0.6, s=25, edgecolors='none')
    
    grid = np.linspace(data_range[0], data_range[1], 200)
    Xm, Ym = np.meshgrid(grid, grid)
    pts = np.column_stack([Xm.ravel(), Ym.ravel()])
    
    # Plot consensus boundary on the left
    if phi_fn is not None and wb is not None:
        Phi_grid = phi_fn(pts)
        wb = np.array(wb)
        if wb.ndim == 1:
            wb_mean = wb
        else:
            wb_mean = np.mean(wb, axis=0)
            
        Z_mean = (Phi_grid @ wb_mean[:-1] + wb_mean[-1]).reshape(Xm.shape)
        ax_left.contour(Xm, Ym, Z_mean, levels=[0], colors='#2F4F4F', linestyles='-.', linewidths=2.5)
        ax_left.plot([], [], color='#2F4F4F', linestyle='-.', linewidth=2.5, label='Consensus Boundary')
        
    ax_left.set_title("Global View (Task 1.2 style)", fontweight='bold')
    ax_left.set_xlabel(r"$x_1$")
    ax_left.set_ylabel(r"$x_2$")
    ax_left.set_aspect("equal")
    ax_left.legend(loc="upper right", framealpha=0.9)
    
    # ── RIGHT PLOT: Agent-specific details (Multi-Agent Partition) ───────────────────
    ax_right = axes[1]
    cmap_agents = plt.get_cmap("Set2")
    
    # Scatter points colored by agent, using class markers
    for i, agent in enumerate(agents_data):
        X_i = agent["X"]
        labels_i = agent["labels"]
        color = cmap_agents(i)
        
        # Label each agent exactly once in the legend
        label_text = f"Agent {i+1}"
        
        # Plot Class +1 for agent i
        pos_idx = (labels_i == 1)
        if np.any(pos_idx):
            ax_right.scatter(X_i[pos_idx, 0], X_i[pos_idx, 1],
                             color=color, marker='^', alpha=0.8, s=35, edgecolors='none',
                             label=label_text)
            label_text = ""  # Prevent duplicate labeling
                             
        # Plot Class -1 for agent i
        neg_idx = (labels_i == -1)
        if np.any(neg_idx):
            ax_right.scatter(X_i[neg_idx, 0], X_i[neg_idx, 1],
                             color=color, marker='v', alpha=0.8, s=35, edgecolors='none',
                             label=label_text)
                             
    # Plot individual agent boundaries and consensus boundary on the right
    if phi_fn is not None and wb is not None:
        Phi_grid = phi_fn(pts)
        if wb.ndim == 2:
            N = wb.shape[0]
            # Draw each agent's local boundary in its corresponding agent color
            for i in range(N):
                wb_i = wb[i]
                Z_i = (Phi_grid @ wb_i[:-1] + wb_i[-1]).reshape(Xm.shape)
                color = cmap_agents(i)
                ax_right.contour(Xm, Ym, Z_i, levels=[0], colors=[color], linestyles='--', linewidths=1.5, alpha=0.8)
            
            # Proxy line for agent boundaries in legend
            ax_right.plot([], [], color='gray', linestyle='--', linewidth=1.5, label='Agent boundaries')
            
            # Draw the consensus boundary (mean of agent weights)
            wb_mean = np.mean(wb, axis=0)
            Z_mean = (Phi_grid @ wb_mean[:-1] + wb_mean[-1]).reshape(Xm.shape)
            ax_right.contour(Xm, Ym, Z_mean, levels=[0], colors='#2F4F4F', linestyles='-.', linewidths=2.5)
            ax_right.plot([], [], color='#2F4F4F', linestyle='-.', linewidth=2.5, label='Consensus Boundary')
            
        elif wb.ndim == 1:
            # If only a single boundary is provided (e.g. true boundary), plot it
            Z = (Phi_grid @ wb[:-1] + wb[-1]).reshape(Xm.shape)
            ax_right.contour(Xm, Ym, Z, levels=[0], colors='#2F4F4F', linestyles='-.', linewidths=2.5)
            ax_right.plot([], [], color='#2F4F4F', linestyle='-.', linewidth=2.5, label='Decision Boundary')
            
    ax_right.set_title("Multi-Agent Data Split & Boundaries", fontweight='bold')
    ax_right.set_xlabel(r"$x_1$")
    ax_right.set_ylabel(r"$x_2$")
    ax_right.set_aspect("equal")
    
    # Add proxy class markers for the right plot legend
    ax_right.scatter([], [], marker='^', color='gray', label='+1', edgecolors='none')
    ax_right.scatter([], [], marker='v', color='gray', label='-1', edgecolors='none')
    
    # Place legend to the right of the right plot
    ax_right.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", framealpha=0.9, fontsize=8)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_task_1_3_boundary_comparison(agents_data, phi_fn, wb_true, wb_learned, map_name="", data_range=(-2, 2)):
    """
    Overlays the true decision boundary and the learned consensus decision boundary
    on the same plot containing the scattered data points.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Boundary Comparison – True vs Learned ({map_name})", fontsize=13, fontweight="bold")
    
    # Aggregate data points to plot in the background
    X_all = []
    labels_all = []
    for agent in agents_data:
        X_all.append(agent["X"])
        labels_all.append(agent["labels"])
    
    X = np.vstack(X_all)
    labels = np.concatenate(labels_all)
    
    # Scatter points
    ax.scatter(X[labels == 1, 0], X[labels == 1, 1], color='#8A2BE2', marker='^', label='+1', alpha=0.6, s=25, edgecolors='none')
    ax.scatter(X[labels == -1, 0], X[labels == -1, 1], color='#FF8C00', marker='v', label='-1', alpha=0.6, s=25, edgecolors='none')
    
    # Grid for drawing the contours
    grid = np.linspace(data_range[0], data_range[1], 200)
    Xm, Ym = np.meshgrid(grid, grid)
    pts = np.column_stack([Xm.ravel(), Ym.ravel()])
    Phi_grid = phi_fn(pts)
    
    # 1. Plot True Boundary 
    Z_true = (Phi_grid @ wb_true[:-1] + wb_true[-1]).reshape(Xm.shape)
    ax.contour(Xm, Ym, Z_true, levels=[0], colors='#2F4F4F', linestyles='-.', linewidths=2.5)
    ax.plot([], [], color='#2F4F4F', linestyle='-.', linewidth=2.5, label='True Boundary (Real)')
    
    # 2. Plot Learned Boundary 
    wb_learned = np.array(wb_learned)
    if wb_learned.ndim == 2:
        wb_learned_mean = np.mean(wb_learned, axis=0)
    else:
        wb_learned_mean = wb_learned
        
    Z_learned = (Phi_grid @ wb_learned_mean[:-1] + wb_learned_mean[-1]).reshape(Xm.shape)
    ax.contour(Xm, Ym, Z_learned, levels=[0], colors='#4169E1', linestyles='-', linewidths=2.5)
    ax.plot([], [], color='#4169E1', linestyle='-', linewidth=2.5, label='Learned Consensus Boundary')

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig


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

def plot_task2_1_trajectories(scenario, title=None, subsample=5):
    """
    2-D visualisation of robot paths, private targets, barycenter trajectory
    and final optimal positions.
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
    
    # Dynamically set the title based on the experiment loop
    if title:
        fig.suptitle(f"{title}\nTrajectory: {lbl}", fontsize=13, fontweight="bold")
    else:
        fig.suptitle(rf"Task 2.1 – Robot Trajectories  |  {lbl}", fontsize=13, fontweight="bold")

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

def plot_task2_1_comparison(scenarios, title="Task 2.1 – Scenario Comparison"):
    """
    Overlays the convergence metrics of multiple scenarios in one figure,
    following the same style as plot_combined_results() in plot_utils.py.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)
    
    # Use the dynamic title passed from main.py
    fig.suptitle(title, fontsize=14, fontweight="bold")

    specs = [
        (axes[0, 0], "cost",        r"Global Cost $J(z, \sigma)$",           False),
        (axes[0, 1], "grad_norm",   r"Gradient Norm $\|\nabla J\|$ (Log)",    True),
        (axes[1, 0], "sigma_error", r"$\sigma$ Estimation Error (Log)",        True),
        (axes[1, 1], "consensus",   r"Consensus Error $\|z - \bar{z}\|$ (Log)", True),
    ]

    for ax, key, metric_title, use_log in specs:
        for sc in scenarios:
            lbl  = sc.get("label", "")
            data = sc["metrics"][key]
            if use_log:
                ax.semilogy(data, linewidth=2, label=f"{lbl}")
            else:
                ax.plot(data,    linewidth=2, label=f"{lbl}")
        
        ax.set_title(metric_title)
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
    
