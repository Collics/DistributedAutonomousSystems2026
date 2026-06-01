# src/task2/task2/scenario_config.py
import numpy as np

import networkx as nx
# ==========================================
# 1. CORE PARAMETERS
# ==========================================
NN       = 6
maxK     = 400
stepsize = 0.01
OBSTACLES = False  # Set to False for Task 2.2, True for Task 2.3
# Task 2.3 specific parameters
d_safe    = 1.0
gamma_cbf = 0.5
obstacles = [0.0, 2.0,
             2.0, -2.0] if OBSTACLES else []
# ==========================================
# 2. TASK 2.2 EXPERIMENT SELECTION
# ==========================================
# EXPERIMENT selects which dimension to vary (mirrors Task 2.1):
#   1 = Parameter Tuning  (vary gamma / beta, fixed geometry and graph)
#   2 = Target Geometry   (vary target shape, fixed params and graph)
#   3 = Network Topology  (vary graph type,   fixed params and geometry)
EXPERIMENT   = 1
# SCENARIO_IDX selects the specific scenario within the experiment:
#   Exp 1:  0 = Balanced (γ=2.0, β=2.0)
#           1 = High Cohesion (γ=0.1, β=2.0)
#           2 = Target Drive  (γ=5.0, β=0.1)
#   Exp 2:  0 = Hexagon
#           1 = Triangle
#           2 = Line
#   Exp 3:  0 = Cycle Graph
#           1 = Path Graph
#           2 = Star Graph
SCENARIO_IDX = 0
# --- Experiment definitions ---
_EXP1 = [
    dict(gamma=2.0, beta=2.0, label='Balanced (\u03b3=2.0, \u03b2=2.0)'),
    dict(gamma=0.1, beta=2.0, label='High Cohesion (\u03b3=0.1, \u03b2=2.0)'),
    dict(gamma=5.0, beta=0.1, label='Target Drive (\u03b3=5.0, \u03b2=0.1)'),
]
_EXP2 = [
    dict(shape='hexagon',  label='Hexagon Target'),
    dict(shape='triangle', label='Triangle Target'),
    dict(shape='line',     label='Line Target'),
]
_EXP3 = [
    dict(graph='cycle', label='Cycle Graph'),
    dict(graph='path',  label='Path Graph'),
    dict(graph='star',  label='Star Graph'),
]
# --- Base (fixed) values used when a dimension is not being varied ---
_BASE_GAMMA = 2.0
_BASE_BETA  = 2.0
_BASE_SHAPE = 'hexagon'
_BASE_GRAPH = 'cycle'
# --- Resolve active scenario ---
if EXPERIMENT == 1:
    _s            = _EXP1[SCENARIO_IDX]
    gamma         = _s['gamma']
    beta          = _s['beta']
    _active_shape = _BASE_SHAPE
    _active_graph = _BASE_GRAPH
    scenario_label = f'Exp.1 \u2013 {_s["label"]}'
elif EXPERIMENT == 2:
    _s            = _EXP2[SCENARIO_IDX]
    gamma         = _BASE_GAMMA
    beta          = _BASE_BETA
    _active_shape = _s['shape']
    _active_graph = _BASE_GRAPH
    scenario_label = f'Exp.2 \u2013 {_s["label"]}'
else:  # EXPERIMENT == 3
    _s            = _EXP3[SCENARIO_IDX]
    gamma         = _BASE_GAMMA
    beta          = _BASE_BETA
    _active_shape = _BASE_SHAPE
    _active_graph = _s['graph']
    scenario_label = f'Exp.3 \u2013 {_s["label"]}'
# ==========================================
# 3. ENVIRONMENT GEOMETRY
# ==========================================
# --- Initial Positions (safe random cluster, matches Task 2.1 style) ---
def _gen_init(N, box=5.0, min_d=0.5, ox=-10.0, seed=42):
    rng = np.random.default_rng(seed)
    z0  = np.zeros((N, 2))
    for i in range(N):
        while True:
            c = rng.uniform(0, box, 2) + np.array([ox, 0.0])
            if i == 0 or np.all(np.linalg.norm(z0[:i] - c, axis=1) >= min_d):
                z0[i] = c
                break
    return z0
Z_init = _gen_init(NN)
# --- Target Positions (geometry matches Task 2.1) ---
def _gen_targets(N, shape, scale=3.0, center=np.array([6.0, 6.0])):
    t = np.zeros((N, 2))
    if shape == 'hexagon':
        for i in range(N):
            a = i * (2 * np.pi / N)
            t[i] = center + scale * np.array([np.cos(a), np.sin(a)])
    elif shape == 'triangle':
        v1 = center + scale * np.array([0.0,  1.0])
        v2 = center + scale * np.array([-np.sqrt(3)/2, -0.5])
        v3 = center + scale * np.array([ np.sqrt(3)/2, -0.5])
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        for i in range(N):
            ei   = i % 3
            step = (i // 3) / max(1, N // 3)
            s, e = edges[ei]
            t[i] = s * (1 - step) + e * step
    elif shape == 'line':
        xs = np.linspace(center[0] - scale, center[0] + scale, N)
        for i in range(N):
            t[i] = np.array([xs[i], center[1]])
    return t
R_targets = _gen_targets(NN, _active_shape)
# ==========================================
# 4. COMMUNICATION GRAPH
# ==========================================
if _active_graph == 'cycle':
    G = nx.cycle_graph(NN)
elif _active_graph == 'path':
    G = nx.path_graph(NN)
else:  # star
    G = nx.star_graph(NN - 1)
Adj = nx.adjacency_matrix(G).toarray()
weightedAdj = np.zeros((NN, NN))
for i in range(NN):
    N_i   = np.nonzero(Adj[i])[0]
    deg_i = len(N_i)
    for j in N_i:
        N_j   = np.nonzero(Adj[j])[0]
        deg_j = len(N_j)
        weightedAdj[i, j] = 1.0 / (1 + max(deg_i, deg_j))
weightedAdj += np.eye(NN) - np.diag(weightedAdj.sum(axis=0))