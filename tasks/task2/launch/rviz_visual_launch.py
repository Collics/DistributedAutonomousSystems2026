from launch import LaunchDescription
from launch_ros.actions import Node

import os
from ament_index_python.packages import get_package_share_directory
import task2.task_config as cfg

'''
This launch file is designed for visualizing ROS 2 bag files that were recorded during Task 2.2 or Task 2.3 simulations.

To use this launch file:
1. launch either
        ros2 launch task2 task2_2_launch.py
        ros2 launch task2 task2_3_launch.py
    to record a bag file of the simulation.
    
2. launch this file specifying the task (defaults to 2.3 if left blank):
        ros2 launch task2 rviz_visual_launch.py 

3. in another terminal, play the bag file you recorded with:
        ros2 bag play <your_bag_file_name>
'''


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory('task2'), 'config.rviz')

    # Base parameters that ALL tasks need
    viz_params = {
        "NN": cfg.NN,
        "plot_title": "ROS 2 Bag Playback",
        "save_name": "playback_data.npy"
    }
    
    for ii in range(cfg.NN):
        viz_params[f"target_{ii}"] = cfg.R_targets[ii].tolist()

    # ONLY add obstacle parameters if they actually exist in the config!
    if len(cfg.obstacles) > 0:
        viz_params["obstacles"] = cfg.obstacles
        viz_params["d_safe"] = cfg.d_safe

    return LaunchDescription([
        Node(
            package="task2",
            executable="central_visualizer",
            name="central_viz",
            parameters=[viz_params],
            output="screen",
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
        )
    ])