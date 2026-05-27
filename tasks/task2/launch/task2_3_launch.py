from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    NN = 6         
    maxK = 1000     
    stepsize = 0.01
    rviz_config = os.path.join(
    get_package_share_directory('task2'),
    'config.rviz'
    )
    # Task 2.3 Scenario Settings
    d_safe = 1.0
    gamma_cbf = 0.5
    obstacles = [0.0, 2.0, 2.0, -2.0] # Flattened for ROS 2 parameters: [[0, 2], [3, -1]]

    # 1. Circular Starting Positions + Random Noise
    np.random.seed(42)
    init_center = np.array([-7.0, 0.0])
    init_radius = 1.0
    Z_init = np.zeros((NN, 2))
    for i in range(NN):
        angle = 2.0 * np.pi * i / NN
        base_pos = init_center + np.array([init_radius * np.cos(angle), init_radius * np.sin(angle)])
        noise = np.random.uniform(-0.5, 0.5, 2) # Adding the random jitter around start position
        Z_init[i] = base_pos + noise

    # 2. Hexagonal Target Positions
    target_center = np.array([7.0, 0.0])
    target_radius = 4.0
    R_targets = np.zeros((NN, 2))
    for i in range(NN):
        angle = 2.0 * np.pi * i / NN
        R_targets[i] = target_center + np.array([target_radius * np.cos(angle), target_radius * np.sin(angle)])

    # 3. Graph Generation: CYCLE GRAPH (not star)
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

    node_list = []
    package_name = "task2" 

    # Global Visualizer Parameters Dictionary
    viz_params = {"NN": NN, "obstacles": obstacles, "d_safe": d_safe}

    # Generate Agent Nodes
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0].tolist()
        weights_ii = [weightedAdj[ii, j] for j in N_ii]
        self_weight = weightedAdj[ii, ii]
        
        viz_params[f"target_{ii}"] = R_targets[ii].tolist()

        node_list.append(
            Node(
                package=package_name,
                namespace=f"agent_{ii}",
                executable="task2_3_agent",
                parameters=[{
                    "id": ii,
                    "stepsize": stepsize,
                    "maxK": maxK,
                    "gamma": 1.0, 
                    "beta": 0.1,   
                    "neighbors": N_ii,
                    "weights": weights_ii,
                    "self_weight": float(self_weight),
                    "r": R_targets[ii].tolist(),
                    "xzero": Z_init[ii].tolist(),
                    "obstacles": obstacles,
                    "d_safe": d_safe,
                    "gamma_cbf": gamma_cbf
                }],
                output="screen",
            )
        )

    # Launch Central Visualizer Node
    node_list.append(
        Node(
            package=package_name,
            executable="central_visualizer",
            name="central_viz",
            parameters=[viz_params],
            output="screen",
        )
    )

    # Record all data via ros2 bag automatically
    node_list.append(
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-a'],
            output='screen'
        )
    )
    
    node_list.append(
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
        )
    )

    return LaunchDescription(node_list)