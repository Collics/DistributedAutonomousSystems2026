
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
TASK_1_1_PLOT_SUMMARY = True

###############
# TASK 1.2 
###############
TASK_1_2 = False

TASK_1_2_M = 500   # total dataset size

TASK_1_2_RANGE = (-2, 2)  # range for generating data
TASK_1_2_STEPSIZE = 5e-4   # step size
TASK_1_2_MAX_ITER = 1000

# Parabola Mapping Parameters (phi_parabola = [x1, x2, x1^2].T)
# Dimension q = 3
W_PARABOLA = np.random.uniform(low=-2.0, high=2.0, size=3).tolist()
B_PARABOLA = float(np.random.uniform(low=-2.0, high=2.0))

# Hyperbola Mapping Parameters (phi_hyperbola = [x1, x2, x1·x2].T) ---
# Dimension q = 3
W_HYPERBOLA = np.random.uniform(low=-2.0, high=2.0, size=3).tolist()
B_HYPERBOLA = float(np.random.uniform(low=-2.0, high=2.0))

###############
# TASK 1.3
###############
TASK_1_3 = True

TASK_1_3_GROUP_NUMBER = 6          # group number (determines P)
TASK_1_3_N        = 5          # number of agents
TASK_1_3_M_LIST   = [500, 1000]   # dataset sizes to test
TASK_1_3_STEPSIZE = 0.005
TASK_1_3_MAX_ITER = 1000
TASK_1_3_RANGE     = (-2, 2)   # range for generating data


###############
# TASK 2.1
###############
RUN_TASK_2_1 = False
TASK_2_1_ANIMATE = True

TASK_2_1_N        = 6          # number of robots
TASK_2_1_ALPHA    = 0.01       # step size
TASK_2_1_MAX_ITER = 1000      # iterations
TASK_2_1_GRAPH    = "cycle"    # 'cycle' | 'path' | 'star'
TASK_2_1_ANIMATE  = True       # show animation (can be slow)

###############
# TASK 2.3
###############
RUN_TASK_2_3 = True

TASK_2_3_N = 6                # Number of agents
TASK_2_3_MAX_ITER = 500      # Iterations
TASK_2_3_ALPHA = 0.01         # Step size (alpha)
TASK_2_3_GAMMA = 1.0          # Weight for private target tracking
TASK_2_3_BETA = 0.1           # Weight for formation tracking (equivalent to lambda in your snippet)

TASK_2_3_GRAPH = "cycle"      # 'cycle' | 'path' | 'star'
TASK_2_3_GAMMA_CBF = 0.5     # CBF safety parameter
TASK_2_3_D_SAFE = 1.0         # Safety distance from obstacle centers
TASK_2_3_OBSTACLES = [[0.0, 2.0], [1.0, -2.0]]  # Center coordinates of obstacles
TASK_2_3_PLOTS = True
TASK_2_3_ANIMATION = True