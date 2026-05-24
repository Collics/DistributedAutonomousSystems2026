###############
# TASK 1.1 
###############
TASK_1_1 = True

TASK_1_1_Path = False
TASK_1_1_Star = True
TASK_1_1_Cycle = False

TASK_1_1_N = 5
TASK_1_1_STEPSIZE = 2 * 1e-1
TASK_1_1_MAX_ITER = 1000

###############
# TASK 1.2 
###############
TASK_1_2 = True

TASK_1_2_M = 500   # total dataset size
TASK_1_2_STEPSIZE = 5e-4   # step size
TASK_1_2_MAX_ITER = 3000

###############
# TASK 1.3
###############
TASK_1_3 = True

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
TASK_2_1_ANIMATE = True
 
TASK_2_1_N        = 6          # number of robots
TASK_2_1_ALPHA    = 0.01       # step size
TASK_2_1_MAX_ITER = 15000      # iterations
TASK_2_1_GRAPH    = "cycle"    # 'cycle' | 'path' | 'star'
TASK_2_1_ANIMATE  = True       # show animation (can be slow)