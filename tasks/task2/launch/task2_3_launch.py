import shutil
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
import task2.task_config as cfg
import numpy as np
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown


def generate_launch_description():
    """Launch file for Task 2.3: Distributed CBF-QP with Obstacle Avoidance."""
    
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(launch_dir, '..', 'task2', 'data'))
    os.makedirs(data_dir, exist_ok=True) # Automatically creates the folder if it doesn't exist
    
    # Assign the absolute path to the bag
    bag_name = os.path.join(data_dir, 'task2_3_bag')

    node_list = []
    package_name = "task2" 

    # All these variables (NN, obstacles, d_safe) come automatically from task_config.py
    viz_params = {"NN": cfg.NN,
                  "obstacles": cfg.obstacles,
                  "d_safe": cfg.d_safe,
                  "plot_title": "Real-Time CBF-QP",
                  "save_name": os.path.join(data_dir, "task2_3_simulation_data.npy")}

    for ii in range(cfg.NN):
        N_ii = np.nonzero(cfg.Adj[ii])[0].tolist()
        weights_ii = [cfg.weightedAdj[ii, j] for j in N_ii]
        self_weight = cfg.weightedAdj[ii, ii]
        
        viz_params[f"target_{ii}"] = cfg.R_targets[ii].tolist()

        node_list.append(
            Node(
                package=package_name,
                namespace=f"agent_{ii}",
                executable="task2_3_agent",
                parameters=[{
                    "id": ii,
                    "stepsize": cfg.stepsize,
                    "maxK": cfg.maxK,
                    "gamma": cfg.gamma,  
                    "beta": cfg.beta,      
                    "neighbors": N_ii,
                    "weights": weights_ii,
                    "self_weight": float(self_weight),
                    "r": cfg.R_targets[ii].tolist(),
                    "xzero": cfg.Z_init[ii].tolist(),
                    "obstacles": cfg.obstacles,
                    "d_safe": cfg.d_safe,
                    "gamma_cbf": cfg.gamma_cbf
                }],
                output="screen",
            )
        )

    # Launch Central Visualizer Node
    viz_node = Node(
        package=package_name,
        executable="central_visualizer",
        name="central_viz",
        parameters=[viz_params],
        output="screen"
    )
    node_list.append(viz_node)

    # NEW: When the visualizer shuts down, kill the entire launch process!
    node_list.append(
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=viz_node,
                on_exit=[EmitEvent(event=Shutdown())]
            )
        )
    )

    
    # Delete the old bag folder if it already exists
    if os.path.exists(bag_name):
        print(f"Overwriting old bag: {bag_name}")
        shutil.rmtree(bag_name)
    
    node_list.append(
        ExecuteProcess(cmd=['ros2', 'bag', 'record',
                            '-a', '-o', bag_name],
                            output='screen')
    )

    return LaunchDescription(node_list)