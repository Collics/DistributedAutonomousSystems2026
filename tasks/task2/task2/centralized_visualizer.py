import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from visualization_msgs.msg import Marker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import task2.task_config as cfg
import sys
import matplotlib.animation as animation

class CentralVisualizer(Node):
    """ROS2 Node that subscribes to all agent topics, visualizes their states in real-time, and generates final convergence plots at the end of the simulation."""
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
        x, y = float(msg.data[2]), float(msg.data[3])
        self.agent_positions[agent_id] = np.array([x, y])
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
            
            if current_k >= cfg.maxK - 1:
                self.finished = True
                self.get_logger().info(f"Target of {cfg.maxK} iterations reached! Auto-generating final plots...")
                self.save_data()
        except Exception:
            pass

    def _create_and_save_animation(self, z_hist, max_k):
        self.get_logger().info("Rendering trajectory animation... (This might take a moment)")
        
        fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
        ax_anim.set_xlim(-12, 12)
        ax_anim.set_ylim(-10, 10)
        ax_anim.set_aspect('equal')
        ax_anim.set_title(f"Trajectory Animation - {self.plot_title}")
        ax_anim.grid(True, linestyle=":", alpha=0.6)

        if self.obstacles.size > 0:
            for obs in self.obstacles:
                ax_anim.add_patch(plt.Circle(obs, self.d_safe, color='tab:orange', alpha=0.3))
                ax_anim.plot(obs[0], obs[1], 'rx')

        for i in range(self.NN):
            target = self.targets[i]
            ax_anim.plot(target[0], target[1], 'bx', markersize=10)

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i / max(self.NN - 1, 1)) for i in range(self.NN)]
        
        robots_scatter = ax_anim.scatter(z_hist[0, :, 0], z_hist[0, :, 1], c=colors, s=80, zorder=5, edgecolors='k')
        tails = [ax_anim.plot([], [], color=colors[i], alpha=0.5, linewidth=1.5)[0] for i in range(self.NN)]

        def init():
            robots_scatter.set_offsets(np.empty((0, 2)))
            for tail in tails:
                tail.set_data([], [])
            return [robots_scatter] + tails

        def update(frame):
            robots_scatter.set_offsets(z_hist[frame])
            tail_length = 30
            start_idx = max(0, frame - tail_length)
            for i in range(self.NN):
                tails[i].set_data(z_hist[start_idx:frame+1, i, 0], z_hist[start_idx:frame+1, i, 1])
            return [robots_scatter] + tails

        skip = max(1, max_k // 200)
        frames = np.arange(0, max_k, skip)
        if frames[-1] != max_k - 1:
            frames = np.append(frames, max_k - 1)

        anim = animation.FuncAnimation(fig_anim, update, frames=frames, init_func=init, blit=True, interval=50)
        
        anim_path = self.save_name.replace('.npy', '.gif')
        try:
            anim.save(anim_path, writer='pillow', fps=20)
            self.get_logger().info(f"[SUCCESS] Animation securely saved to: {anim_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save animation: {e}")
        
        plt.close(fig_anim)

    def save_data(self):
        raw_history = np.array(self.history)
        np.save(self.save_name, raw_history)
        
        plt.ioff()
        plt.close(self.fig) 
        
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

        self._create_and_save_animation(z_hist, max_k)

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

        # NEW DASHBOARD LAYOUT (Mirrors Python exactly)
        fig = plt.figure(figsize=(14, 16))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3)
        fig.suptitle(f"ROS 2 Convergence Metrics  |  Iters: {max_k-1}", fontsize=16, fontweight="bold")

        # --- ROW 1: Trajectories (Spans both columns) ---
        ax_traj = fig.add_subplot(gs[0, :])
        cmap = plt.get_cmap("tab10")
        
        if self.obstacles.size > 0:
            for obs in self.obstacles:
                ax_traj.add_patch(plt.Circle(obs, self.d_safe, color='tab:red', alpha=0.3))
                ax_traj.plot(obs[0], obs[1], 'rx')

        for i in range(self.NN):
            color = cmap(i / max(self.NN - 1, 1))
            ax_traj.plot(z_hist[:, i, 0], z_hist[:, i, 1], color=color, lw=2.5, label=f'R{i}')
            ax_traj.plot(z_hist[0, i, 0], z_hist[0, i, 1], 'o', color=color)
            ax_traj.plot(self.targets[i][0], self.targets[i][1], 'bx', markersize=10, markeredgewidth=2)
            
        ax_traj.set_title("Trajectory Deflection", fontsize=14, fontweight='bold')
        ax_traj.set_aspect('equal')
        ax_traj.grid(True, alpha=0.5)
        ax_traj.legend(loc='upper right', fontsize=10)

        # --- ROW 2: Consensus Metrics ---
        ax_sx = fig.add_subplot(gs[1, 0])
        ax_sy = fig.add_subplot(gs[1, 1])
        for i in range(self.NN):
            color = cmap(i / max(self.NN - 1, 1))
            ax_sx.plot(s_hist[:, i, 0], color=color, alpha=0.8, label=f'R{i}')
            ax_sy.plot(s_hist[:, i, 1], color=color, alpha=0.8)

        ax_sx.set_title(r"Barycenter Estimate Consensus ($s_x$)", fontsize=12, fontweight='bold')
        ax_sx.set_xlabel("Iteration $k$")
        ax_sx.grid(True, alpha=0.5)
        ax_sx.legend(loc='best', fontsize=8)

        ax_sy.set_title(r"Barycenter Estimate Consensus ($s_y$)", fontsize=12, fontweight='bold')
        ax_sy.set_xlabel("Iteration $k$")
        ax_sy.grid(True, alpha=0.5)

        # --- ROW 3: Convergence Metrics ---
        ax_cost = fig.add_subplot(gs[2, 0])
        ax_cost.plot(cost, lw=2.5, color='tab:blue')
        ax_cost.set_title(r"Global Cost $J(z, \sigma)$", fontsize=12, fontweight='bold')
        ax_cost.set_yscale('log')
        ax_cost.set_xlabel("Iteration $k$")
        ax_cost.grid(True, alpha=0.5)

        ax_grad = fig.add_subplot(gs[2, 1])
        ax_grad.plot(grad_norm, lw=2.5, color='tab:orange')
        ax_grad.set_title(r"Gradient Norm $\|\nabla J\|$", fontsize=12, fontweight='bold')
        ax_grad.set_yscale('log')
        ax_grad.set_xlabel("Iteration $k$")
        ax_grad.grid(True, alpha=0.5)

        fig.subplots_adjust(top=0.92)
        
        image_path = self.save_name.replace('.npy', '.png')
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        
        self.get_logger().info(f"[SUCCESS] High-res dashboard securely saved to: {image_path}")
        self.get_logger().info("--> Auto-closing in 2 seconds to continue batch... <--")
        
        plt.show(block=False)
        plt.pause(2.0)
        sys.exit(0)
        
def main(args=None):
    rclpy.init(args=args)
    viz = CentralVisualizer()
    try:
        rclpy.spin(viz)
    except SystemExit:
        pass 
    except KeyboardInterrupt:
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