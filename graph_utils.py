import networkx as nx
import numpy as np

def get_graph_and_matrix(n_agents, graph_type):
    """
    Generates a networkx graph and its corresponding Metropolis-Hastings 
    weighted adjacency matrix. Includes connectivity check.
    """
    IN = np.eye(n_agents)
    
    # 1. Generate graph and check for connectivity using matrix power
    while True:
        if graph_type == 'cycle':
            G = nx.cycle_graph(n_agents)
        elif graph_type == 'path':
            G = nx.path_graph(n_agents)
        elif graph_type == 'star':
            # nx.star_graph(n) generates 1 center + n outer nodes = n+1 total nodes.
            G = nx.star_graph(n_agents - 1)
        elif graph_type == 'random':
            # Binomial/Erdos-Renyi graph
            G = nx.binomial_graph(n=n_agents, p=0.4)
        elif graph_type == 'complete':
            # Complete graph with n_agents nodes
            G = nx.complete_graph(n_agents)
        else:
            raise ValueError(f"Topology {graph_type} not supported.")

        # Check connectivity using matrix power
        Adj = nx.adjacency_matrix(G).toarray()
        test = np.linalg.matrix_power(Adj + IN, n_agents)
        
        if np.all(test > 0):
            print(f"The {graph_type} graph is connected.")
            break
        else:
            print(f"The {graph_type} graph is not connected, trying again...")

    # Initialize the weighted adjacency matrix
    A = np.zeros((n_agents, n_agents))
    
    # 2. Compute Metropolis-Hastings weights using NetworkX properties
    # Iterate only over existing edges (no need for inner loops to find them!)
    for i, j in G.edges():
        # w_ij = 1 / (1 + max(degree_i, degree_j))
        weight = 1.0 / (1.0 + max(G.degree[i], G.degree[j]))
        A[i, j] = weight
        A[j, i] = weight

    # 3. Vectorized diagonal computation - Compute self-loop weights (diagonal entries)
    A = A + IN - np.diag(np.sum(A, axis=0)) # Add the identity and subtract the diagonal matrix of column sums to ensure row-stochasticity
    # for i in range(n_agents):
    #     A[i, i] = 1.0 - np.sum(A[i, :])

    return G, A