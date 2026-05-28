#
# Distributed Autonomous Systems
# Task 1.1 - Distributed Optimization
# Ivan Colangelo, Nicholas Gioia, Alexandru Zaporojanu
# Bologna, 09/06/26
#

import networkx as nx
import numpy as np
import Parameters as par
from graph_utils import get_graph_and_matrix
from plots import plot_task_1_1_summary_results, plot_task_1_1_network, plot_task_1_1_consensus_dynamics

def compute_metrics_at_k(z_k):
    """
    Computes the global Cost Function, the norm of the global Gradient,
    and the Consensus Error, all evaluated at the mean consensus point z_bar.
    """
    # Mean consensus point across all agents
    z_bar = np.mean(z_k, axis=0)

    total_cost  = 0.0
    global_grad = np.zeros(par.d)

    for i in range(par.TASK_1_1_N):
        # global cost at z_bar
        total_cost += 0.5 * z_bar.T @ par.Q[i] @ z_bar + par.r[i].T @ z_bar
        # global gradient at z_bar
        global_grad += par.Q[i] @ z_bar + par.r[i]

    # norm of the global gradient
    grad_norm = np.linalg.norm(global_grad)

    # consensus error: distance of each agent's estimate from z_bar
    consensus_err = np.linalg.norm(z_k - z_bar)

    return total_cost, grad_norm, consensus_err

#  Main task function
def task1_1():
    print("\n" + "="*50)
    print("--- Task 1.1: Distributed Optimization ---")
    print("="*50 + "\n")

    d = par.d
    N = par.TASK_1_1_N
    stepsize = par.TASK_1_1_STEPSIZE
    maxK = par.TASK_1_1_MAX_ITER
    Q = par.Q
    r = par.r

    all_costs = {}
    all_grads = {}
    all_consensus = {}
    all_graphs = {}

     # z* = -(sum(Q_i))^-1 * sum(r_i)
    sum_Q = np.sum(Q, axis=0)
    sum_r = np.sum(r, axis=0)
    z_star_1_1 = -np.linalg.inv(sum_Q) @ sum_r

    for gt in par.GRAPH_TYPES:
        print(f"\n ======== [Task 1.1] Testing graph type: {gt} ========")
        G, A = get_graph_and_matrix(N, gt)

        all_graphs[gt] = (G)
        print(f"  Adjacency matrix:\n{A}")
        print(np.round(A, 2))

        #check column-stochasticity
        col_sums = np.sum(A, axis=0)
        print(f"  Column sums: {col_sums}")

        # check row-stochasticity
        row_sums = np.sum(A, axis=1)
        print(f"  Row sums: {row_sums}")

        # check if the matrix is doubly-stochastic
        if np.isclose(col_sums, 1.0).all() and np.isclose(row_sums, 1).all():
            print("  The weight matrix is doubly-stochastic.")
        
        # Alghorithm
        z = np.zeros((maxK, N, d))
        s = np.zeros((maxK, N, d))

        cost_hist = np.zeros(maxK)
        grad_norm_hist = np.zeros(maxK)
        consensus_hist = np.zeros(maxK)

        # Initialization
        z[0] = np.random.randn(N, d) * 5 # z[0] contain values in [-5, 5]

        for ii in range(N):
            s[0, ii] = Q[ii] @ z[0, ii] + r[ii] # grad of local cost at initialization
        
        # Compute initial metrics at k=0
        cost_hist[0], grad_norm_hist[0], consensus_hist[0] = compute_metrics_at_k(z[0]) 

        # Main loop
        for k in range(maxK - 1):
            for ii in range(N):
                N_ii = list(G.neighbors(ii)) # indices of neighbors of ii
                
                # --- Update decision variable z ---
                z[k+1, ii] = A[ii, ii] * z[k, ii]

                # contribution from neighbors
                for jj in N_ii:
                    z[k+1, ii] += A[ii, jj] * z[k, jj]
                
                # gradient step
                z[k+1, ii] -= stepsize * s[k, ii]

                # Compute new gradient at z[k+1, ii]
                grad_new = Q[ii] @ z[k+1, ii] + r[ii] # grad of local cost at new point
                grad_current = Q[ii] @ z[k, ii] + r[ii] # grad of local cost at current point

                # --- Update gradient tracking variable s ---
                s[k+1, ii] = A[ii, ii] * s[k, ii]

                # contribution from neighbors
                for jj in N_ii:
                    s[k+1, ii] += A[ii, jj] * s[k, jj]

                # add gradient correction
                s[k+1, ii] += grad_new - grad_current
            
            # Compute metrics for plotting and evaluation
            cost_hist[k + 1], grad_norm_hist[k + 1], consensus_hist[k + 1] = compute_metrics_at_k(z[k + 1])
        
        all_costs[gt] = cost_hist
        all_grads[gt] = grad_norm_hist
        all_consensus[gt] = consensus_hist

        print(f"  - Final cost: {cost_hist[-1]:.4f}")
        print(f"  - Final gradient norm: {grad_norm_hist[-1]:.2e}")
        print(f"  - Final consensus error: {consensus_hist[-1]:.2e}")

        if par.TASK_1_1_PLOT_CONSENSUS:
            plot_task_1_1_consensus_dynamics(z, gt, z_star = z_star_1_1)
    
    if par.TASK_1_1_PLOT_NETWORK:
        plot_task_1_1_network(all_graphs)
    
    if par.TASK_1_1_PLOT_SUMMARY:
        plot_task_1_1_summary_results(all_costs, all_grads, all_consensus)
        
            
