import numpy as np
import networkx as nx

import Parameters as par
from tasks.task1_1_DEF import build_metropolis_weights
from tasks.task1_2_DEF import phi_parabola, logistic_grad, logistic_loss, generate_dataset, misclassification_rate

# ──────────────────────────────────────────────────────────────
#  Dataset splitting
# ──────────────────────────────────────────────────────────────
def split_dataset_even_groups(X, y, N_agents, P, seed=42):
    """
    Divide the dataset for even-group agents.
 
    - P% of data → Baseline Random (split equally among agents).
    - (100-P)% → sorted by feature x2 and split into contiguous blocks.
    """

    rng = np.random.default_rng(seed)
    M = len(X)

    # 1. Calcolo percentuale P
    num_baseline = int(M * (P / 100))
    
    # 2. Creiamo un array di indici da 0 a M-1 e lo mescoliamo
    indices       = np.arange(M)
    rng.shuffle(indices)

    idx_baseline  = indices[:num_baseline]
    idx_remaining = indices[num_baseline:]

    # Dividiamo gli indici della baseline equamente tra gli N agenti
    baseline_splits = np.array_split(idx_baseline, N_agents)

    # 3. Vertical Feature-Biased Split sul (100-P)% rimanente
    # Estraiamo i valori della feature x2 (seconda colonna, indice 1) per i dati rimanenti
    x2_remaining = X[idx_remaining, 1]

    # np.argsort restituisce l'ordine degli indici per avere l'array crescente
    sorted_order = np.argsort(x2_remaining)
    idx_remaining_sorted = idx_remaining[sorted_order]

    # Dividiamo gli indici ordinati in N blocchi contigui
    sorted_splits = np.array_split(idx_remaining_sorted, N_agents)

    # 4. Uniamo le due parti per ciascun agente
    agent_data = {}
    for i in range(N_agents):
        idx = np.concatenate((baseline_splits[i], sorted_splits[i]))
        agent_data[i] = {"X": X[idx], "y": y[idx]}
 
    return agent_data

_GRAPH_BUILDERS = {
    1: ("path",  lambda N: nx.path_graph(N)),
    2: ("star",  lambda N: nx.star_graph(N - 1)),
    3: ("cycle", lambda N: nx.cycle_graph(N)),
}

def run_single(M_size: int, graph_id: int, N: int, P: float,
                stepsize: float, max_iter: int):
    graph_name, graph_fn = _GRAPH_BUILDERS[graph_id]
    G  = graph_fn(N)
    W  = build_metropolis_weights(G)
 
    print(f"\n  [Task 1.3] graph={graph_name} | M={M_size} | N={N} | "
          f"P={P:.0f}% | α={stepsize}")
 
    # Dataset
    rng    = np.random.default_rng(99)
    w_true = rng.standard_normal(3)   # 3 features for Parabola
    b_true = rng.uniform(-0.5, 0.5)
    X, y   = generate_dataset(M_size, w_true, b_true, phi_parabola)
 
    agents = split_dataset_even_groups(X, y, N, P)
    for i in range(N):
        agents[i]["Phi"] = phi_parabola(agents[i]["X"])
 
    # Initialisation
    d        = 4               # q=3 features + 1 bias
    z        = np.zeros((N, d))
    s        = np.zeros((N, d))
    grad_old = np.zeros((N, d))
 
    for i in range(N):
        grad_old[i] = logistic_grad(z[i], agents[i]["Phi"], agents[i]["y"])
        s[i]        = grad_old[i].copy()
 
    cost_hist      = []
    grad_norm_hist = []
    consensus_hist = []
 
    # Gradient tracking loop
    for k in range(max_iter):
        z_new    = W @ z - stepsize * s
        grad_new = np.zeros((N, d))
        total_cost = 0.0
 
        for i in range(N):
            grad_new[i]  = logistic_grad(z_new[i], agents[i]["Phi"], agents[i]["y"])
            total_cost  += logistic_loss(z_new[i], agents[i]["Phi"], agents[i]["y"])
 
        s_new    = W @ s + grad_new - grad_old
        z        = z_new
        s        = s_new
        grad_old = grad_new
 
        cost_hist.append(total_cost)
        grad_norm_hist.append(np.linalg.norm(grad_new))
        consensus_hist.append(np.linalg.norm(z - np.mean(z, axis=0)))
 
    # Evaluation
    final_wb = np.mean(z, axis=0)
    Phi_all  = phi_parabola(X)
    miss     = misclassification_rate(final_wb, Phi_all, y)
 
    print(f"    Final total loss:     {cost_hist[-1]:.4f}")
    print(f"    Final gradient norm:  {grad_norm_hist[-1]:.2e}")
    print(f"    Consensus error:      {consensus_hist[-1]:.2e}")
    print(f"    Misclassification:    {miss:.2f}%")
 
    return {
        "graph_name":      graph_name,
        "M":               M_size,
        "cost_history":    cost_hist,
        "grad_norm_history": grad_norm_hist,
        "consensus_history": consensus_hist,
        "miss_rate":       miss,
        "title": f"Task 1.3 – {graph_name.capitalize()} graph | M={M_size}",
    }

# ──────────────────────────────────────────────────────────────
#  Main task function
# ──────────────────────────────────────────────────────────────
 
def task1_3():
    """
    Run Task 1.3 for all combinations of dataset size × graph topology
    defined in Parameters.py.
 
    Returns
    -------
    results : list of dicts (one per combination), each with keys:
        'graph_name', 'M', 'cost_history', 'grad_norm_history',
        'consensus_history', 'miss_rate', 'title'
    """
    group_id = par.TASK_1_3_GROUP_NUMBER
    N        = par.TASK_1_3_N
    P        = 40 + (group_id % 3) * 10 
    stepsize = par.TASK_1_3_STEPSIZE
    max_iter = par.TASK_1_3_MAX_ITER
    M_list   = par.TASK_1_3_M_LIST
    graphs   = par.TASK_1_3_GRAPHS
 
    print(f"\n[Task 1.3] Group {group_id}: P={P}% | N={N} agents")
 
    results = []
    for graph_id in graphs:
        for M_size in M_list:
            res = run_single(M_size, graph_id, N, P, stepsize, max_iter)
            results.append(res)
 
    return results