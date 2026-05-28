# src/task2/task2/scenario_config.py
import numpy as np
import networkx as nx

# ==========================================
# 1. CORE PARAMETERS
# ==========================================
NN = 6         
maxK = 1000     
stepsize = 0.01

# Task 2.3 specific parameters
d_safe = 1.0
gamma_cbf = 0.5
obstacles = [0.0, 2.0, 2.0, -2.0] # Flat list for ROS parameters

# ==========================================
# 2. ENVIRONMENT GEOMETRY
# ==========================================
# Starting Positions (Circle with noise)
np.random.seed(42)
init_center = np.array([-7.0, 0.0])
init_radius = 1.0
Z_init = np.zeros((NN, 2))
for i in range(NN):
    angle = 2.0 * np.pi * i / NN
    base_pos = init_center + np.array([init_radius * np.cos(angle), init_radius * np.sin(angle)])
    Z_init[i] = base_pos + np.random.uniform(-0.5, 0.5, 2)

# Target Positions (Hexagon)
target_center = np.array([7.0, 0.0])
target_radius = 4.0
R_targets = np.zeros((NN, 2))
for i in range(NN):
    angle = 2.0 * np.pi * i / NN
    R_targets[i] = target_center + np.array([target_radius * np.cos(angle), target_radius * np.sin(angle)])

# ==========================================
# 3. COMMUNICATION GRAPH
# ==========================================
G = nx.cycle_graph(NN)
Adj = nx.adjacency_matrix(G).toarray()

weightedAdj = np.zeros((NN, NN)) 
for i in range(NN):
    N_i = np.nonzero(Adj[i])[0]
    deg_i = len(N_i)
    for j in N_i:
        N_j = np.nonzero(Adj[j])[0]
        deg_j = len(N_j)
        weightedAdj[i, j] = 1.0 / (1 + max(deg_i, deg_j))

weightedAdj += np.eye(NN) - np.diag(weightedAdj.sum(axis=0))