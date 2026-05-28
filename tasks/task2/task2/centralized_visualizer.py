import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from visualization_msgs.msg import Marker
import numpy as np
import matplotlib.pyplot as plt

class CentralVisualizer(Node):
    def __init__(self):
        super().__init__('central_visualizer', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        self.NN = self.get_parameter('NN').value
        
        #  fetch obstacles (Default to empty array if missing)
        obs_param = self.get_parameter('obstacles')
        if obs_param.value is not None:
            self.obstacles = np.array(obs_param.value).reshape(-1, 2)
        else:
            self.obstacles = np.array([]) 

        # fetch d_safe (Default to 0.0 if missing)
        d_safe_param = self.get_parameter('d_safe')
        self.d_safe = d_safe_param.value if d_safe_param.value is not None else 0.0
        
        # Dynamic Titles and File Names
        self.plot_title = self.get_parameter_or('plot_title', rclpy.Parameter('plot_title', rclpy.Parameter.Type.STRING, 'Real-Time Tracking')).value
        self.save_name = self.get_parameter_or('save_name', rclpy.Parameter('save_name', rclpy.Parameter.Type.STRING, 'simulation_data.npy')).value
        
        self.agent_positions = {i: np.zeros(2) for i in range(self.NN)}
        self.targets = {i: np.array(self.get_parameter(f'target_{i}').value) for i in range(self.NN)}
        self.history = []

        # RViz Publisher
        self.rviz_pub = self.create_publisher(Marker, '/visualization_topic', 10)
        self.create_timer(1.0, self.publish_static_env)
        # Passive Subscriptions
        for i in range(self.NN):
            self.create_subscription(MsgFloat, f'/topic_{i}', self.listener_callback, 10)

        # Setup Matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.timer = self.create_timer(0.05, self.update_visuals) 

    def publish_static_env(self):
        """Draws the obstacles and targets in RViz before the bag starts."""
        # 1. Publish Obstacles
        if self.obstacles.size > 0:
            for idx, obs in enumerate(self.obstacles):
                self.rviz_pub.publish(self._make_marker(
                    idx, 'obstacles', obs[0], obs[1], Marker.CYLINDER, [1.0, 0.5, 0.0, 0.5], self.d_safe * 2.0
                ))
                
        # 2. Publish Targets
        for i in range(self.NN):
            self.rviz_pub.publish(self._make_marker(
                i + 100, 'targets', self.targets[i][0], self.targets[i][1], Marker.CUBE, [1.0, 0.0, 0.0, 1.0], 0.3
            ))
    
    def listener_callback(self, msg):
        agent_id = int(msg.data[0])
        x, y = float(msg.data[2]), float(msg.data[3])
        
        self.agent_positions[agent_id] = np.array([x, y])
        self.history.append([int(msg.data[1]), agent_id, x, y])

        # Publish Agent instantly (Zero jitter)
        self.rviz_pub.publish(self._make_marker(
            agent_id + 200, 'agents', x, y, Marker.SPHERE, [1.0, 0.0, 1.0, 1.0], 0.4
        ))

    def _make_marker(self, marker_id, ns, x, y, shape, color, scale=0.4):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = marker_id
        m.type = shape
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = float(scale)
        m.scale.y = float(scale)
        m.scale.z = 0.1 if shape == Marker.CYLINDER else float(scale)
        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    def update_visuals(self):
        # --- 1. Update Matplotlib ---
        self.ax.clear()
        
        # Draw obstacles ONLY if they exist (Task 2.3)
        if self.obstacles.size > 0:
            for obs in self.obstacles:
                self.ax.add_patch(plt.Circle(obs, self.d_safe, color='tab:orange', alpha=0.3))
                self.ax.plot(obs[0], obs[1], 'rx')

        # Draw Targets and Agents
        for i in range(self.NN):
            self.ax.plot(self.targets[i][0], self.targets[i][1], 'bx', markersize=10)
            self.ax.plot(self.agent_positions[i][0], self.agent_positions[i][1], 'go', markersize=8)

        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.set_title(self.plot_title)  # Dynamic Title
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_data(self):
        np.save(self.save_name, np.array(self.history))
        print(f"Simulation data saved to '{self.save_name}'.")

def main(args=None):
    rclpy.init(args=args)
    viz = CentralVisualizer()
    try:
        rclpy.spin(viz)
    except KeyboardInterrupt:
        viz.save_data()
    finally:
        viz.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()