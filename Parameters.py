"""
Parameters 
Ivan Colangelo, Nicholas Gioia, Alexandru Zaporojanu
Bologna, 09/06/26
"""
import numpy as np

###############
# TASK 1.1 
###############
TASK_1_1 = False

GRAPH_TYPES = ['star', 'cycle', 'path', 'complete', 'random'] # all graph types to test

d = 2
TASK_1_1_N = 5
TASK_1_1_STEPSIZE = 1e-2
TASK_1_1_MAX_ITER = 1000


# Local quadratic cost function: l_i(z) = 0.5 * z^T * Q_i * z + r_i^T * z
Q = np.zeros((TASK_1_1_N, d, d))
r = np.zeros((TASK_1_1_N, d))

for i in range(TASK_1_1_N):
    temp = np.random.randn(d, d) 
    Q[i] = 0.5 * (temp + temp.T) + 2 * np.eye(d) 
    r[i] = 0 * (np.random.rand(d) - 0.5) 

# Flags for plotting
TASK_1_1_PLOT_CONSENSUS = True
TASK_1_1_PLOT_NETWORK = True
TASK_1_1_PLOT_SINGLE_RESULTS = True
TASK_1_1_PLOT_SUMMARY = True

###############
# TASK 1.2 
###############
TASK_1_2 = False

TASK_1_2_M = 500   # total dataset size

TASK_1_2_RANGE = (-2, 2)  # range for generating data
TASK_1_2_STEPSIZE = 5e-4   # step size
TASK_1_2_MAX_ITER = 1000
TASK_1_2_RANDOM_MAPPING = False 

# Parabola Mapping Parameters (phi_parabola = [x1, x2, x1^2].T)
W_PARABOLA = [0.0, -1.0, 1.0] 
B_PARABOLA = - 1.0

# Dimension q = 3
W_HYPERBOLA = [0.0, 0.0, 1.0] 
B_HYPERBOLA = - 0.5

if TASK_1_2_RANDOM_MAPPING:
    # Dimension q = 3
    W_PARABOLA = np.random.uniform(low=-2.0, high=2.0, size=3).tolist()
    B_PARABOLA = float(np.random.uniform(low=-2.0, high=2.0))

    # Hyperbola Mapping Parameters (phi_hyperbola = [x1, x2, x1·x2].T) ---
    # Dimension q = 3
    W_HYPERBOLA = np.random.uniform(low=-2.0, high=2.0, size=3).tolist()
    B_HYPERBOLA = float(np.random.uniform(low=-2.0, high=2.0))

# Flags for plotting
TASK_1_2_FUTURE_MAPPING = True
TASK_1_2_FUTURE_MAPPING_COMPARISON = True
TASK_1_2_METRICS = True

###############
# TASK 1.3
###############
TASK_1_3 = False

TASK_1_3_GROUP_NUMBER = 6          # group number (determines P)
TASK_1_3_N        = 5          # number of agents
TASK_1_3_M_LIST   = [500, 1500]   # dataset sizes to test
TASK_1_3_STEPSIZE = 0.005
TASK_1_3_MAX_ITER = 1000
TASK_1_3_RANGE     = (-2, 2)   # range for generating data

# Flags for plotting
TASK_1_3_DATA_SPLIT = True
TASK_1_3_FUTURE_MAPPING = True
TASK_1_3_FUTURE_MAPPING_COMPARISON = True
TASK_1_3_PLOT_CONSENSUS = True
TASK_1_3_PLOT_SINGLE_RESULTS = True
TASK_1_3_METRICS = True

###############
# TASK 2.1
###############
RUN_TASK_2_1 = True

TASK_2_1_N        = 6          # Number of robots
TASK_2_1_ALPHA    = 0.01       # Step size
TASK_2_1_MAX_ITER = 1000       # Iterations
TASK_2_1_EXP_DIR  = "figs/Task2_1_Experiments"  # Output directory for plots and GIFs

# --- EXPERIMENT 1: Parameter Tuning Sets ---
# List of tuples (gamma, beta) to test against each other
TASK_2_1_PARAM_SETS = [(2.0, 2.0), (0.1, 2.0), (5.0, 0.1)]
TASK_2_1_PARAM_LABELS = [
    "Balanced (γ=2.0, β=2.0)", 
    "High Cohesion (γ=0.1, β=2.0)", 
    "Target Drive (γ=5.0, β=0.1)"
]
TASK_2_1_PARAM = (2.0, 2.0)  # Fixed (gamma, beta) to use for other experiments (geometry tests)


# --- EXPERIMENT 2: Geometry Tests ---
# The shapes to test
TASK_2_1_SHAPES = ['hexagon', 'triangle', 'line']
TASK_2_1_SHAPE = 'hexagon'  # Fixed shape to use for other experiments (parameter tuning)

# --- EXPERIMENT 3: Network Topologies ---
# The graph types to test
TASK_2_1_GRAPHS = ['cycle', 'path', 'star']
TASK_2_1_GRAPH = 'cycle'  # Fixed graph type to use for other experiments (parameter tuning)

# --- Experiment Selection Flags ---
# Set to True to run that experiment, False to skip it
TASK_2_1_RUN_EXP1 = True   # Experiment 1: Parameter Tuning
TASK_2_1_RUN_EXP2 = False   # Experiment 2: Target Geometries
TASK_2_1_RUN_EXP3 = False   # Experiment 3: Network Topologies

# Animation Flag
TASK_2_1_ANIMATE = True


###############
# TASK 2.3
###############
RUN_TASK_2_3 = False

TASK_2_3_N = 6                # Number of agents
TASK_2_3_MAX_ITER = 500       # Iterations
TASK_2_3_ALPHA = 0.01         # Step size (alpha)
TASK_2_3_D_SAFE = 1.0         # Safety distance from obstacle centers

TASK_2_3_EXP_DIR = "figs/Task2_3_Experiments"  # Output directory for plots and GIFs
TASK_2_3_ANIMATE = True

# Fixed Tracking Parameters across ALL Task 2.3 experiments
TASK_2_3_PARAM = (2.0, 0.1)  # (gamma, beta) 

# --- EXPERIMENT 1: CBF Parameter Tuning (gamma_cbf) ---
TASK_2_3_EXP1_GCBF = [0.1, 1.0, 4.0]
TASK_2_3_EXP1_LABELS = ["Conservative", "Balanced", "Aggressive"]
TASK_2_3_GCBF = 1.0  # Fixed CBF param to use for other experiments (geometry & graph)

# --- EXPERIMENT 2: Obstacle Geometries ---
TASK_2_3_EXP2_OBSTACLES = [
    [[0.0, 2.0], [2.0, -2.0]],                   
    [[0.0, 0.0]],                                
    [[2.0, 1.5], [2.0, -1.5], [2.0, 0.0]]        
]
TASK_2_3_EXP2_LABELS = ["Standard", "Center_Block", "Wall"]
TASK_2_3_OBSTACLES = [[0.0, 2.0], [2.0, -2.0]] # Fixed obstacle layout for other experiments

# --- EXPERIMENT 3: Network Topologies ---
TASK_2_3_EXP3_GRAPHS = ['cycle', 'path', 'star']
TASK_2_3_EXP3_LABELS = ["Cycle", "Path", "Star"]
TASK_2_3_GRAPH = 'cycle'  # Fixed graph type for other experiments