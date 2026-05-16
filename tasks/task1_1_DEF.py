#
# DAS
# Task 1.1
# Ivan Colangelo, Nicholas Gioia, Alexandru Zaporojanu
# Bologna, 09/06/26
#

import networkx as nx
import numpy as np
import Parameters as par

def _build_graph(graph_type: str, N: int) -> nx.Graph:
    """Return a NetworkX graph given a type string."""
    gt = graph_type.lower()
    if gt == "path":
        return nx.path_graph(N)
    elif gt == "star":
        return nx.star_graph(N - 1)
    elif gt == "cycle":
        return nx.cycle_graph(N)
    else:
        raise ValueError(f"Unknown graph type: '{graph_type}'. "
                         "Choose from 'path', 'star', 'cycle'.")
 
def build_metropolis_weights(G: nx.Graph) -> np.ndarray:
    """
    Compute the Metropolis-Hastings weight matrix for graph G.
    Returns a doubly-stochastic (N x N) numpy array.
    """
    N   = G.number_of_nodes()
    Adj = nx.adjacency_matrix(G).toarray()
    weightedAdj  = np.zeros((N, N))
 
    for i in range(N):
        N_i   = np.nonzero(Adj[i])[0]
        deg_i = len(N_i)
        for j in N_i:
            N_j   = np.nonzero(Adj[j])[0]
            deg_j = len(N_j)
            weightedAdj[i, j] = 1.0 / (1 + max(deg_i, deg_j))
 
    weightedAdj += np.eye(N) - np.diag(weightedAdj.sum(axis=0))
    return weightedAdj

def _cost_function(z, Q, r):
    val  = 0.5 * Q * z * z + r * z
    grad = Q * z + r
    return val, grad

#  Main task function
def task1_1(graph_type: str = None):
    N = par.TASK_1_1_N
    stepsize = par.TASK_1_1_STEPSIZE
    maxK = par.TASK_1_1_MAX_ITER

    # Graph selection
    if graph_type is None:
        if par.TASK_1_1_Path:
            graph_type = "path"
        elif par.TASK_1_1_Star:
            graph_type = "star"
        elif par.TASK_1_1_Cycle:
            graph_type = "cycle"
        else:
            raise ValueError("No graph flag set in Parameters.py for Task 1.1.")
        
    G = _build_graph(graph_type, N)
    Adj = nx.adjacency_matrix(G).toarray()
    weightedAdj = build_metropolis_weights(G)

    # Random quadratic parameters
    np.random.seed(0)
    Q = np.random.rand(N) + 1          # uniform in [1, 2]
    r = 10 * (np.random.rand(N) - 0.5) # uniform in [-5, 5]

    # Algorithm
    z = np.zeros((maxK, N))
    s = np.zeros((maxK, N))
    for ii in range(N):
        _, g = _cost_function(z[0, ii], Q[ii], r[ii])
        s[0, ii] = g
 
    cost      = np.zeros(maxK)
    gradient  = np.zeros(maxK)
    consensus = np.zeros(maxK)
 
    for k in range(maxK - 1):
        for ii in range(N):
            N_ii = np.nonzero(Adj[ii])[0]
 
            z[k+1, ii] = weightedAdj[ii, ii] * z[k, ii]
            for jj in N_ii:
                z[k+1, ii] += weightedAdj[ii, jj] * z[k, jj]
            z[k+1, ii] -= stepsize * s[k, ii]
 
            _, g_new = _cost_function(z[k+1, ii], Q[ii], r[ii])
            ell_ii, g_old = _cost_function(z[k, ii], Q[ii], r[ii])
 
            s[k+1, ii] = weightedAdj[ii, ii] * s[k, ii]
            for jj in N_ii:
                s[k+1, ii] += weightedAdj[ii, jj] * s[k, jj]
            s[k+1, ii] += g_new - g_old
 
            cost[k]     += ell_ii
            gradient[k] += g_old
 
        consensus[k] = np.linalg.norm(z[k] - np.mean(z[k]))
 
    print(f"  Final cost: {cost[-2]:.4f} | Final |grad|: {abs(gradient[-2]):.2e}")
 
    metrics = {"cost": cost, "gradient": gradient, "consensus": consensus}
    return weightedAdj, metrics
