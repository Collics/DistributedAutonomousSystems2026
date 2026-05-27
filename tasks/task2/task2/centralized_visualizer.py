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
        obs_flat = self.get_parameter('obstacles').value
        self.obstacles = np.array(obs_flat).reshape(-1, 2)
        self.d_safe = self.get_parameter('d_safe').value
        
        self.agent_positions = {i: np.zeros(2) for i in range(self.NN)}
        self.targets = {i: np.array(self.get_parameter(f'target_{i}').value) for i in range(self.NN)}
        self.history = []

        # RViz Publisher
        self.rviz_pub = self.create_publisher(Marker, '/visualization_topic', 10)

        # Passive Subscriptions
        for i in range(self.NN):
            self.create_subscription(MsgFloat, f'/topic_{i}', self.listener_callback, 10)

        # Setup Matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        # Slow down the Matplotlib refresh rate slightly to match the slowed agents
        self.timer = self.create_timer(0.05, self.update_visuals) 

    def listener_callback(self, msg):
        agent_id = int(msg.data[0])
        x, y = float(msg.data[2]), float(msg.data[3])
        self.agent_positions[agent_id] = np.array([x, y])
        self.history.append([int(msg.data[1]), agent_id, x, y])

    def _make_marker(self, marker_id, ns, x, y, shape, color, scale=0.4):
        """Helper to create RViz markers exactly like Task 2.2"""
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
        if shape == Marker.CYLINDER:
            m.scale.z = 0.1 # Flat cylinder for obstacles
        else:
            m.scale.z = float(scale)
        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    def update_visuals(self):
        """Updates both Matplotlib and RViz in real-time"""
        # --- 1. Update Matplotlib ---
        self.ax.clear()
        
        for obs in self.obstacles:
            self.ax.add_patch(plt.Circle(obs, self.d_safe, color='tab:orange', alpha=0.3))
            self.ax.plot(obs[0], obs[1], 'rx')

        for i in range(self.NN):
            self.ax.plot(self.targets[i][0], self.targets[i][1], 'bx', markersize=10)
            self.ax.plot(self.agent_positions[i][0], self.agent_positions[i][1], 'go', markersize=8)

        self.ax.set_xlim(-12, 12) 
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.set_title("Real-Time CBF-QP Tracking")
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # --- 2. Update RViz ---
        # Draw Obstacles (Orange Cylinders)
        for idx, obs in enumerate(self.obstacles):
            # Scale is diameter, so 2 * d_safe
            self.rviz_pub.publish(self._make_marker(idx, 'obstacles', obs[0], obs[1], Marker.CYLINDER, [1.0, 0.5, 0.0, 0.5], self.d_safe * 2.0))
            
        # Draw Targets (Red Cubes)
        for i in range(self.NN):
            self.rviz_pub.publish(self._make_marker(i + 100, 'targets', self.targets[i][0], self.targets[i][1], Marker.CUBE, [1.0, 0.0, 0.0, 1.0], 0.3))

        # Draw Agents (Purple Spheres)
        for i in range(self.NN):
            self.rviz_pub.publish(self._make_marker(i + 200, 'agents', self.agent_positions[i][0], self.agent_positions[i][1], Marker.SPHERE, [1.0, 0.0, 1.0, 1.0], 0.4))

    def save_data(self):
        np.save('task2_3_simulation_data.npy', np.array(self.history))
        print("Simulation data saved to 'task2_3_simulation_data.npy'.")

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