#
# Distributed Autonomous Systems
# Task 1.3 - Distributed Classification
# Ivan Colangelo, Nicholas Gioia, Alexandru Zaporojanu
# Bologna, 09/06/26
#

import numpy as np
import networkx as nx
import Parameters as par
from graph_utils import get_graph_and_matrix
from tasks.task1_2_DEF import centralized_gradient_descent, phi_parabola, phi_hyperbola, generate_dataset
from plots import plot_task_1_1_consensus_dynamics, plot_task1_3_data_split, plot_task_1_3_individual_distr_metrics, plot_task_1_3_metrics, plot_task1_3_dataset_boundary, plot_task_1_3_boundary_comparison
# ──────────────────────────────────────────────────────────────
#  Dataset splitting
# ──────────────────────────────────────────────────────────────
def split_dataset_even_groups(X, labels):
    """
    Divide the dataset for even-group agents.
 
    - P% of data → Baseline Random (split equally among agents).
    - (100-P)% → sorted by feature x2 and split into contiguous blocks.
    """

    rng = np.random.default_rng(seed=0)
    M = len(X)
    G = par.TASK_1_3_GROUP_NUMBER
    N = par.TASK_1_3_N

    P_percent = 40 + (G % 3) * 10
    M_baseline = int(M * (P_percent / 100)) # number of samples for baseline random split

    # Baseline Random Split
    indices = np.arange(M) # array of indices from 0 to M-1
    rng.shuffle(indices) # shuffle the indices randomly

    idx_baseline  = indices[:M_baseline] # first P% for baseline random split
    idx_remaining = indices[M_baseline:] # remaining (100-P)% for feature-biased split

    # Split the baseline indices into N equal parts for the random split
    baseline_splits = np.array_split(idx_baseline, N)

    x2_remaining = X[idx_remaining, 1] # extract the x2 feature for the remaining data
    sorted_order = np.argsort(x2_remaining) # get the order of indices to sort by x2

    sorted_idx = idx_remaining[sorted_order] # sort the remaining indices by x2

    # Split the sorted indices into N contiguous blocks for the feature-biased split
    sorted_splits = np.array_split(sorted_idx, N)

    # Combine the baseline and sorted splits for each agent
    agent_datasets = []
    for i in range(N):
        idx_agent = np.concatenate((baseline_splits[i], sorted_splits[i])) # combine baseline and sorted indices for agent i
        agent_datasets.append({
            "X": X[idx_agent],
            "labels": labels[idx_agent]
        })
    return agent_datasets


# Helper function

def single_agent_logistic_grad(z_i, agent_data_i, phi_fn):
    """
    Computes the local gradient of the logistic loss for a single agent.
    z_i          : (q+1,)
    agent_data_i : dict with 'X' and 'labels'
    phi_func     : callable R^d -> R^q
    """

    X_i      = agent_data_i['X']
    labels_i = agent_data_i['labels']
 
    # Augment: [phi(X_i), 1]  →  (M_i, q+1)
    Phi_i        = phi_fn(X_i)                                   # (M_i, q)
    phi_tilde_i  = np.hstack([Phi_i, np.ones((len(X_i), 1))])     # (M_i, q+1)
 
    # Gradient of  sum_m log(1 + exp(-p_m * phi_tilde_m @ z_i))
    exp_term = 1.0 + np.exp(labels_i * (phi_tilde_i @ z_i))       # (M_i,)
    grad_i   = np.sum((-labels_i[:, None] * phi_tilde_i)
                      / exp_term[:, None], axis=0)                 # (q+1,)
    return grad_i

def global_logistic_metrics(z_matrix, X_global, labels_global, phi_fn):
    """
    Evaluates global cost and gradient norm at the consensus mean z_bar.
    z_matrix : (N, q+1)
    X_global  : (M, d) all data points
    labels_global : (M,) all labels
    phi_fn   : feature mapping function
    Returns:
    - total_cost : scalar logistic loss at z_bar
    - grad_norm  : scalar norm of the global gradient at z_bar
    - z_bar      : (q+1,) consensus mean of the agents' parameters
    """
    z_bar = np.mean(z_matrix, axis=0)                              # (q+1,)
 
    Phi_all      = phi_fn(X_global)                              # (M, q)
    phi_tilde    = np.hstack([Phi_all,
                               np.ones((len(X_global), 1))])       # (M, q+1)
 
    # Cost: numerically stable with logaddexp
    argument   = -labels_global * (phi_tilde @ z_bar)
    total_cost = np.sum(np.logaddexp(0, argument))
 
    # Gradient norm
    exp_term    = 1.0 + np.exp(labels_global * (phi_tilde @ z_bar))
    global_grad = np.sum((-labels_global[:, None] * phi_tilde)
                         / exp_term[:, None], axis=0)
    grad_norm   = np.linalg.norm(global_grad)
 
    return total_cost, grad_norm, z_bar


#  Single experiment run
def run_ev(X, labels, phi_fn, mapping):
    """Runs one experiment for a given dataset and feature mapping.
    X      : (M, d) dataset
    labels : (M,) binary labels
    phi_fn : feature mapping function
    mapping : string name of the mapping (for plotting)
    """
    N = par.TASK_1_3_N
    stepsize = par.TASK_1_3_STEPSIZE
    max_iter = par.TASK_1_3_MAX_ITER
    M = len(X)

    # ──- Centralised baseline (for reference) ─────────────────────────────
    print(f"\n{'-'*55}")
    print(f" MAPPING: {mapping.upper()} | Dataset Size: {M}")
    print(f"{'-'*55}")
    print("  [Baseline] Running Centralised Gradient Descent...")
    theta_centr, cost_centr, grad_centr = centralized_gradient_descent(X, phi_fn, labels, stepsize, max_iter)

    # --- Data Splitting for Distributed Algorithm ────────────────────────────────────────────
    agents = split_dataset_even_groups(X, labels)

    # True boundary parameters from Parameters.py
    if mapping.lower() == "parabola":
        w_true = par.W_PARABOLA
        b_true = par.B_PARABOLA
    else:
        w_true = par.W_HYPERBOLA
        b_true = par.B_HYPERBOLA
    wb_true = np.concatenate([w_true, [b_true]])
    
    # Plot dataset with true boundary
    if par.TASK_1_3_FUTURE_MAPPING:
        plot_task1_3_dataset_boundary(agents, phi_fn, wb=wb_true, map_name=f"{mapping} (True Boundary)", data_range=par.TASK_1_3_RANGE)

    distributed_cost_hist = {}
    distributed_grad_hist = {}

    # ──- Gradient Tracking Loop ─────────────────────────────────────────────
    for gt in par.GRAPH_TYPES:
        G, A = get_graph_and_matrix(N, gt)

        print(f"\n  ➤ TOPOLOGY: {gt.upper()} | N = {N} Agents")
        print("    Running Distributed Gradient Tracking...")

        # ── Initialisation ────────────────────────────────────────────────────
        q_dim = phi_fn(X[0:1]).shape[1] # dimension of the feature mapping output (q)
        d_opt = q_dim + 1 # dimension of the optimization variable [w; b]
        z = np.zeros((max_iter,N, d_opt))
        s = np.zeros((max_iter,N, d_opt))

        cost_hist = np.zeros(max_iter)
        grad_hist = np.zeros(max_iter)

        z[0] = np.random.randn(N, d_opt) # random init
        for ii in range(N):
            s[0, ii] = single_agent_logistic_grad(z[0, ii], agents[ii], phi_fn)
        
        cost_hist[0], grad_hist[0], z_bar = global_logistic_metrics(z[0], X, labels, phi_fn)

        for k in range(max_iter - 1):
            for ii in range(N):
                # Update z[k+1, ii] based on the graph structure
                N_ii = list(G.neighbors(ii))

                z[k+1, ii] = A[ii, ii] * z[k, ii]
                for jj in N_ii:
                    z[k+1, ii] += A[ii, jj] * z[k, jj]
                z[k+1, ii] -= stepsize * s[k, ii]

                # Compute the new gradient at z[k+1, ii] and the old gradient at z[k, ii]
                grad_new = single_agent_logistic_grad(z[k+1, ii], agents[ii], phi_fn)
                grad_curr = single_agent_logistic_grad(z[k, ii], agents[ii], phi_fn)

                # Update s[k+1, ii] using the gradient tracking formula  
                s[k+1, ii] = A[ii, ii] * s[k, ii]
                for jj in N_ii:
                    s[k+1, ii] += A[ii, jj] * s[k, jj]
                s[k+1, ii] += grad_new - grad_curr

            cost_hist[k+1], grad_hist[k+1], z_bar = global_logistic_metrics(z[k+1], X, labels, phi_fn) # for plotting

        distributed_cost_hist[gt] = cost_hist
        distributed_grad_hist[gt] = grad_hist

        if par.TASK_1_3_PLOT_CONSENSUS:
            plot_task_1_1_consensus_dynamics(z, gt, z_star = theta_centr)
        if par.TASK_1_3_PLOT_SINGLE_RESULTS:
            plot_task_1_3_individual_distr_metrics(cost_centr, grad_centr, cost_hist, grad_hist, gt, mapping_name = mapping)
        if par.TASK_1_3_FUTURE_MAPPING:
            plot_task1_3_dataset_boundary(agents, phi_fn, wb=z[-1], map_name=f"{mapping} - Learned ({gt})", data_range=par.TASK_1_3_RANGE)
        if par.TASK_1_3_FUTURE_MAPPING_COMPARISON:
            plot_task_1_3_boundary_comparison(agents, phi_fn, wb_true=wb_true, wb_learned=z[-1], map_name=f"{mapping} ({gt})", data_range=par.TASK_1_3_RANGE)

        print("\n --- Performance Evaluation --- ")
        # --- Centralised baseline results
        w_centr, b_centr = theta_centr[:-1], theta_centr[-1]
        pred_labels_centr = generate_dataset(X, w_centr, b_centr, phi_fn)
        acc_rate_centr = np.mean(pred_labels_centr == labels) * 100
        missclass_centr = M - int((acc_rate_centr / 100) * M)

        # --- Distributed results
        w_distr, b_distr = z_bar[:-1], z_bar[-1]
        pred_labels_distr = generate_dataset(X, w_distr, b_distr, phi_fn)
        acc_rate_distr = np.mean(pred_labels_distr == labels) * 100
        missclass_distr = M - int((acc_rate_distr / 100) * M)

        print("    [Performance Evaluation]")
        print(f"    ├─ Centralised : Accuracy {acc_rate_centr:6.2f}% | Missclassified: {missclass_centr}")
        print(f"    └─ Distributed : Accuracy {acc_rate_distr:6.2f}% | Missclassified: {missclass_distr}")


    if par.TASK_1_3_METRICS:
        plot_task_1_3_metrics(cost_centr, grad_centr, distributed_cost_hist, distributed_grad_hist, mapping)

        

# ──────────────────────────────────────────────────────────────
#  Main task function
# ──────────────────────────────────────────────────────────────
 
def task1_3():
    np.random.seed(0)
    M_list = par.TASK_1_3_M_LIST

    range = par.TASK_1_3_RANGE

    for M in M_list:
        print("\n" + "="*65)
        print(f" [TASK 1.3] EVALUATING DATASET: M = {M} SAMPLES | Range: {range}")
        print("="*65)
        # Generate random points
        lower, upper = range
        X = np.random.uniform(lower, upper, (M, 2)) # M random points in 2D

        labels = np.zeros(M)
        agents_dataset = split_dataset_even_groups(X, labels) # labels for splitting (not used in this form)
        agents_X = [agent_data['X'] for agent_data in agents_dataset]

        if par.TASK_1_3_DATA_SPLIT:
            plot_task1_3_data_split(agents_X, f"Data Split for M={M}")

        labels_parabola = generate_dataset(X, w=par.W_PARABOLA, b=par.B_PARABOLA, phi_fn=phi_parabola)
        labels_hyperbola = generate_dataset(X, w=par.W_HYPERBOLA, b=par.B_HYPERBOLA, phi_fn=phi_hyperbola)

        run_ev(X, labels_parabola, phi_parabola, "Parabola")
        run_ev(X, labels_hyperbola, phi_hyperbola, "Hyperbola")
 