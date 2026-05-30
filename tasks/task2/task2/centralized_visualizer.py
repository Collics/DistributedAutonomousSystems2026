import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from visualization_msgs.msg import Marker
import numpy as np
import matplotlib.pyplot as plt
import task2.task_config as cfg
import sys

class CentralVisualizer(Node):
    def __init__(self):
        super().__init__('central_visualizer', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        self.NN = self.get_parameter('NN').value
        
        obs_param = self.get_parameter('obstacles')
        self.obstacles = np.array(obs_param.value).reshape(-1, 2) if obs_param.value is not None else np.array([]) 

        d_safe_param = self.get_parameter('d_safe')
        self.d_safe = d_safe_param.value if d_safe_param.value is not None else 0.0
        
        self.plot_title = self.get_parameter_or('plot_title', rclpy.Parameter('plot_title', rclpy.Parameter.Type.STRING, 'Real-Time Tracking')).value
        self.save_name = self.get_parameter_or('save_name', rclpy.Parameter('save_name', rclpy.Parameter.Type.STRING, 'playback_data.npy')).value
        
        self.agent_positions = {i: np.array(cfg.Z_init[i]) for i in range(self.NN)}
        self.targets = {i: np.array(self.get_parameter(f'target_{i}').value) for i in range(self.NN)}
        self.history = []
        self.finished = False

        self.rviz_pub = self.create_publisher(Marker, '/visualization_topic', 10)
        self.create_timer(1.0, self.publish_static_env)
        
        for i in range(self.NN):
            self.create_subscription(MsgFloat, f'/topic_{i}', self.listener_callback, 1000)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.timer = self.create_timer(0.05, self.update_visuals) 

    def publish_static_env(self):
        if self.obstacles.size > 0:
            for idx, obs in enumerate(self.obstacles):
                self.rviz_pub.publish(self._make_marker(idx, 'obstacles', obs[0], obs[1], Marker.CYLINDER, [1.0, 0.5, 0.0, 0.5], self.d_safe * 2.0))
        for i in range(self.NN):
            self.rviz_pub.publish(self._make_marker(i + 100, 'targets', self.targets[i][0], self.targets[i][1], Marker.CUBE, [1.0, 0.0, 0.0, 1.0], 0.3))
    
    def listener_callback(self, msg):
        agent_id = int(msg.data[0])
        # msg.data layout: [agent_id, k, zx, zy, sx, sy, vx, vy]
        x, y = float(msg.data[2]), float(msg.data[3])
        self.agent_positions[agent_id] = np.array([x, y])
        
        # Save the full state history for post-simulation analysis (cost, gradient norms, consensus)
        self.history.append(list(msg.data)) 

        self.rviz_pub.publish(self._make_marker(agent_id + 200, 'agents', x, y, Marker.SPHERE, [1.0, 0.0, 1.0, 1.0], 0.4))

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
        try:
            self.ax.clear()
            if self.obstacles.size > 0:
                for obs in self.obstacles:
                    self.ax.add_patch(plt.Circle(obs, self.d_safe, color='tab:orange', alpha=0.3))
                    self.ax.plot(obs[0], obs[1], 'rx')
            for i in range(self.NN):
                self.ax.plot(self.targets[i][0], self.targets[i][1], 'bx', markersize=10)
                self.ax.plot(self.agent_positions[i][0], self.agent_positions[i][1], 'go', markersize=8)

            self.ax.set_xlim(-12, 12)
            self.ax.set_ylim(-10, 10)
            self.ax.set_aspect('equal')
            
            current_k = int(self.history[-1][1]) if self.history else 0
            self.ax.set_title(f"{self.plot_title}  |  Iteration: {current_k}")
            
            self.ax.grid(True)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # when maxK is reached, stop the simulation and auto-generate final plots
            if current_k >= cfg.maxK - 1:
                self.finished = True
                self.get_logger().info(f"Target of {cfg.maxK} iterations reached! Auto-generating final plots...")
                self.save_data()
        except Exception:
            # Ignore GUI crashes during shutdown!
            pass

    def save_data(self):
        raw_history = np.array(self.history)
        np.save(self.save_name, raw_history)
        
        plt.ioff()  # Turn off live interactive mode
        plt.close(self.fig) # Safely destroy the live window
        
        if len(raw_history) == 0: return

        max_k = int(np.max(raw_history[:, 1])) + 1
        z_hist = np.zeros((max_k, self.NN, 2))
        s_hist = np.zeros((max_k, self.NN, 2))
        
        for row in raw_history:
            a_id, k = int(row[0]), int(row[1])
            z_hist[k, a_id] = [row[2], row[3]]
            s_hist[k, a_id] = [row[4], row[5]]

        for k in range(1, max_k):
            for i in range(self.NN):
                if np.array_equal(z_hist[k, i], [0.0, 0.0]):
                    z_hist[k, i] = z_hist[k-1, i]
                if np.array_equal(s_hist[k, i], [0.0, 0.0]):
                    s_hist[k, i] = s_hist[k-1, i]

        cost = np.zeros(max_k)
        grad_norm = np.zeros(max_k)

        for k in range(max_k):
            sigma_k = np.mean(z_hist[k], axis=0)
            c_val, g_val = 0.0, 0.0
            for i in range(self.NN):
                zi = z_hist[k, i]
                ri = cfg.R_targets[i]
                c_val += cfg.gamma * np.dot(zi - ri, zi - ri) + cfg.beta * np.dot(zi - sigma_k, zi - sigma_k)
                g1 = 2.0 * cfg.gamma * (zi - ri) + 2.0 * cfg.beta * (zi - sigma_k)
                g_val += np.linalg.norm(g1)**2
            cost[k] = c_val
            grad_norm[k] = np.sqrt(g_val)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"ROS 2 Convergence Metrics  |  Iters: {max_k-1}", fontsize=14, fontweight="bold")
        
        axes[0,0].plot(cost, lw=2.5, color='tab:blue')
        axes[0,0].set_title(r"Global Cost $J(z, \sigma)$")
        axes[0,0].set_yscale('log')
        
        axes[0,1].plot(grad_norm, lw=2.5, color='tab:orange')
        axes[0,1].set_title(r"Gradient Norm $\|\nabla J\|$ (Log)")
        axes[0,1].set_yscale('log')

        cmap = plt.get_cmap("tab10")
        for i in range(self.NN):
            color = cmap(i)
            axes[1,0].plot(s_hist[:, i, 0], color=color, alpha=0.8, label=f'R{i}')
            axes[1,1].plot(s_hist[:, i, 1], color=color, alpha=0.8)

        axes[1,0].set_title(r"Barycenter Estimate Consensus ($s_x$)")
        axes[1,1].set_title(r"Barycenter Estimate Consensus ($s_y$)")
        axes[1,0].legend(loc='best', fontsize=8)

        for ax in axes.flat:
            ax.grid(True, which="both", alpha=0.4)
            ax.set_xlabel("Iteration $k$")
            
        plt.tight_layout(h_pad=2.5)
        
        image_path = self.save_name.replace('.npy', '.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        
        self.get_logger().info(f"[SUCCESS] High-res plot securely saved to: {image_path}")
        self.get_logger().info("--> Close the Matplotlib window to exit the simulation! <--")
        
        plt.show(block=True)
        
        # Kill the ROS 2 node cleanly when the user manually closes the graph window
        sys.exit(0)
        

def main(args=None):
    rclpy.init(args=args)
    viz = CentralVisualizer()
    try:
        rclpy.spin(viz)
    except SystemExit:
        pass # Expected exit when the user closes the final plot window
    except KeyboardInterrupt:
        # Fallback: If you hit Ctrl+C early, it intercepts and generates the plot safely anyway!
        if not viz.finished:
            viz.finished = True
            viz.get_logger().info("Simulation stopped early! Generating final metrics...")
            viz.save_data()
    finally:
        if rclpy.ok():
            viz.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()