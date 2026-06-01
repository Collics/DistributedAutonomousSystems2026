from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

import os
from ament_index_python.packages import get_package_share_directory
import task2.task_config as cfg

'''
This launch file visualizes ROS 2 bag files recorded during Task 2.2 or Task 2.3.

To use this launch file:
1. Open src/task2/task2/task_config.py
2. Set TASK_MODE = '2.2' (for nominal) or TASK_MODE = '2.3' (for safety CBF)
3. Rebuild your workspace: colcon build --packages-select task2
4. Launch this file: ros2 launch task2 rviz_visual_launch.py 
'''

def generate_launch_description():
    # Safely locate the rviz config. If it isn't in the install folder, RViz will still open!
    try:
        rviz_config = os.path.join(get_package_share_directory('task2'), 'config.rviz')
        rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []
    except Exception:
        rviz_args = []

    # Dynamically find the src/task2/task2/data folder
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(launch_dir, '..', 'task2', 'data'))
    os.makedirs(data_dir, exist_ok=True)

    # Force the visualizer to save the .npy file inside the data folder
    npy_path = os.path.join(data_dir, 'playback_data.npy')

    viz_params = {
        "NN": cfg.NN,
        "plot_title": f"ROS 2 Bag Playback (Task {cfg.TASK_MODE})",
        "save_name": npy_path 
    }
    
    for ii in range(cfg.NN):
        viz_params[f"target_{ii}"] = cfg.R_targets[ii].tolist()

    # Read config to set obstacles and determine WHICH bag to play
    if hasattr(cfg, 'obstacles') and len(cfg.obstacles) > 0:
        viz_params["obstacles"] = cfg.obstacles
        viz_params["d_safe"] = getattr(cfg, 'd_safe', 1.0)
        bag_name = os.path.join(data_dir, 'task2_3_bag')  # Obstacles exist, must be Task 2.3
    else:
        bag_name = os.path.join(data_dir, 'task2_2_bag')  # No obstacles, must be Task 2.2

    # 1. Visualizer Node
    viz_node = Node(
        package="task2",
        executable="central_visualizer",
        name="central_viz",
        parameters=[viz_params],
        output="screen",
    )
    
    # 2. RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=rviz_args,
        output="screen"
    )

    # 3. Auto-Play Bag (Delayed by 2.5 seconds to let RViz load)
    play_bag = TimerAction(
        period=2.5,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'bag', 'play', bag_name],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        viz_node,
        rviz_node,
        play_bag
    ])