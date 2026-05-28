###############
# TASK 1.1 
###############
TASK_1_1 = False

TASK_1_1_Path = False
TASK_1_1_Star = True
TASK_1_1_Cycle = False

TASK_1_1_N = 5
TASK_1_1_STEPSIZE = 1e-5
TASK_1_1_MAX_ITER = 1000

###############
# TASK 1.2 
###############
TASK_1_2 = False

TASK_1_2_M = 500   # total dataset size
TASK_1_2_STEPSIZE = 5e-1   # step size
TASK_1_2_MAX_ITER = 3000

###############
# TASK 1.3
###############
TASK_1_3 = False

TASK_1_3_GROUP_NUMBER = 6          # group number (determines P)
TASK_1_3_N        = 5          # number of agents
TASK_1_3_M_LIST   = [500, 1000, 1500]   # dataset sizes to test
TASK_1_3_STEPSIZE = 0.05
TASK_1_3_MAX_ITER = 2000
# Graph topologies to test: subset of {1: path, 2: star, 3: cycle}
TASK_1_3_GRAPHS   = [1, 2, 3]

###############
# TASK 2.1
###############
RUN_TASK_2_1 = True
 
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