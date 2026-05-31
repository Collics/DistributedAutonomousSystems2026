from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import numpy as np
import os
import task2.task_config as cfg
import shutil

def generate_launch_description():
    
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(launch_dir, '..', 'task2', 'data'))
    os.makedirs(data_dir, exist_ok=True) # Automatically creates the folder if it doesn't exist
    
    # Assign the absolute path to the bag
    bag_name = os.path.join(data_dir, 'task2_2_bag')
    
        
    node_list = []
    package_name = "task2"
    
    viz_params = {
        "NN": cfg.NN,
        "plot_title": f"Task 2.2 | {cfg.scenario_label}",
        "save_name": os.path.join(data_dir, "task2_2_simulation_data.npy")
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
                    "gamma": cfg.gamma, 
                    "beta": cfg.beta,      
                    "neighbors": N_ii,
                    "weights": weights_ii ,
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