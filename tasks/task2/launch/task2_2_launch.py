from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import numpy as np
import networkx as nx
import os
import datetime
from ament_index_python.packages import get_package_share_directory
import task2.task_config as cfg
import shutil

def generate_launch_description():

    rviz_config = os.path.join(
        get_package_share_directory('task2'),
        'config.rviz'
    )
    
    node_list = []
    package_name = "task2"
    bag_name = 'task2_2_bag'
    
    viz_params = {
        "NN": cfg.NN,
        "plot_title": "Real-Time Aggregative Tracking (Task 2.2)",
        "save_name": "task2_2_simulation_data.npy"
    }

    for ii in range(cfg.NN):
        N_ii = np.nonzero(cfg.Adj[ii])[0].tolist()
        weights_ii = [cfg.weightedAdj[ii, j] for j in N_ii]
        self_weight = cfg.weightedAdj[ii, ii]
        
        viz_params[f"target_{ii}"] = cfg.R_targets[ii].tolist()

        node_list.append(
            Node(
                package=package_name,
                namespace=f"agent_{ii}",
                executable="task2_2_agent",
                parameters=[{
                    "id": ii,
                    "stepsize": cfg.stepsize,
                    "maxK": cfg.maxK,
                    "gamma": 1.0, 
                    "beta": 0.1,   
                    "neighbors": N_ii,
                    "weights": weights_ii,
                    "self_weight": float(self_weight),
                    "r": cfg.R_targets[ii].tolist(),
                    "xzero": cfg.Z_init[ii].tolist()
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
 
    
    # 2. Silently delete the old bag folder if it already exists
    if os.path.exists(bag_name):
        print(f"Overwriting old bag: {bag_name}")
        shutil.rmtree(bag_name)
    
    node_list.append(
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '-a', '-o', bag_name], output='screen')
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