from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

'''
This launch file is designed for visualizing ROS 2 bag files that were recorded during Task 2.2 or Task 2.3 simulations.
It launches the central visualizer node and RViz, allowing you to see the trajectories of the agents as they were recorded
in the bag file.

To use this launch file:
1. launche either
        ros2 launch task2 task2_2_launch.py
        ros2 launch task2 task2_3_launch.py
    to record a bag file of the simulation.
    
2. launch this file with:
        ros2 launch task2 rviz_visual_launch.py

3. in another terminal, play the bag file you recorded with:
        ros2 bag play <your_bag_file_name>
'''


def generate_launch_description():
    NN = 6         
    d_safe = 1.0
    obstacles = [0.0, 2.0, 2.0, -2.0] # Only needed for Task 2.3 bags

    # Hexagonal Target Positions (needed for the visualizer to draw the red cubes)
    target_center = np.array([7.0, 0.0])
    target_radius = 4.0
    R_targets = np.zeros((NN, 2))
    for i in range(NN):
        angle = 2.0 * np.pi * i / NN
        R_targets[i] = target_center + np.array([target_radius * np.cos(angle), target_radius * np.sin(angle)])

    rviz_config = os.path.join(
        get_package_share_directory('task2'),
        'config.rviz'
    )

    viz_params = {
        "NN": NN,
        "obstacles": obstacles,
        "d_safe": d_safe,
        "plot_title": "ROS 2 Bag Playback",
        "save_name": "playback_data.npy"
    }
    for ii in range(NN):
        viz_params[f"target_{ii}"] = R_targets[ii].tolist()

    return LaunchDescription([
        # 1. Launch ONLY the Visualizer
        Node(
            package="task2", # Change to task2_2 if playing a Task 2.2 bag
            executable="central_visualizer",
            name="central_viz",
            parameters=[viz_params],
            output="screen",
        ),
        # 2. Launch RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
        )
    ])