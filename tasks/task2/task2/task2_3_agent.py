import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
import numpy as np
from time import sleep
from scipy.optimize import minimize, LinearConstraint

def local_cost(zi, sigma, r_i, gamma_i, beta_i):
    """Local cost function for agent i."""
    return gamma_i * np.dot(zi - r_i, zi - r_i) + beta_i * np.dot(zi - sigma, zi - sigma)
 
def grad1_li(zi, sigma, r_i, gamma_i, beta_i):
    """Gradient of the local cost with respect to zi."""
    return 2.0 * gamma_i * (zi - r_i) + 2.0 * beta_i * (zi - sigma)
 
def grad2_li(zi, sigma, beta_i):
    """Gradient of the local cost with respect to sigma."""
    return -2.0 * beta_i * (zi - sigma)

def phi(z_i):
    """Identity mapping for the state."""
    return z_i

def grad_phi(z_i):
    """Gradient of the identity mapping is just the identity matrix."""
    return 1.0

# --- CBF-QP Filter ---
def cbf_qp_filter(z_i, u_nom, obstacles, d_safe, gamma_cbf):
    """Applies a CBF-QP filter to ensure safety with respect to obstacles."""
    if len(obstacles) == 0: # No obstacles, return nominal control
        return u_nom

    diff = z_i[None, :] - obstacles # (num_obstacles, 2)
    grad_V = 2.0 * diff # Gradient of the CBF: V = sum((z_i - obs_j)^2) - d_safe^2
    A = -grad_V 
    b = gamma_cbf * (np.sum(diff**2, axis=1) - d_safe**2)

    def objective(u): return np.sum((u - u_nom)**2) # Quadratic cost to stay close to nominal control
    def jacobian(u): return 2 * (u - u_nom) # Gradient of the objective

    constraints = LinearConstraint(A, -np.inf, b)
    res = minimize(objective, np.copy(u_nom), method='SLSQP', jac=jacobian, 
                   constraints=constraints, options={'ftol': 1e-9, 'disp': False}) # Suppress output and set tight tolerance

    if res.success:
        return res.x # Return the optimized control input that satisfies the safety constraints
    else:
        return np.zeros(2) # Emergency stop

class Agent(Node):
    """ROS2 Node representing a single agent in the distributed optimization algorithm."""
    def __init__(self):
        super().__init__("safety_agent", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        # Parameters
        self.stepsize = self.get_parameter("stepsize").value
        self.maxK = self.get_parameter("maxK").value    
        self.gamma = self.get_parameter("gamma").value
        self.beta = self.get_parameter("beta").value
        
        self.r = np.array(self.get_parameter("r").value) 
        self.agent_id = float(self.get_parameter("id").value)
        self.neighbors = self.get_parameter("neighbors").value
        
        # Safety parameters
        obs_flat = self.get_parameter("obstacles").value
        self.obstacles = np.array(obs_flat).reshape(-1, 2) if obs_flat else np.array([])
        self.d_safe = self.get_parameter("d_safe").value
        self.gamma_cbf = self.get_parameter("gamma_cbf").value

        weight = self.get_parameter("weights").value
        self.weights = {n: w for n, w in zip(self.neighbors, weight)}
        self.self_weight = self.get_parameter("self_weight").value 

        # Initialization
        self.k = 0
        self.z = np.array(self.get_parameter("xzero").value)
        self.s = phi(self.z)
        self.v = grad2_li(self.z, self.s, self.beta)

        # Setup Decentralized Communication
        for j in self.neighbors:
            self.create_subscription(MsgFloat,
                                     f"/topic_{j:.0f}",
                                     self.listener_callback,
                                     1000)

        self.received_data = {j: [] for j in self.neighbors}
        self.publisher = self.create_publisher(MsgFloat,
                                               f"/topic_{self.agent_id:.0f}",
                                               1000)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def listener_callback(self, msg):
        j = int(msg.data[0])
        self.received_data[j].append(list(msg.data[1:]))

    def timer_callback(self):
        msg = MsgFloat()
        # First iteration: publish initial state without update
        if self.k == 0:  
            if self.publisher.get_subscription_count() < len(self.neighbors): # Wait until all neighbors are subscribed before publishing
                return
            raw_data = [self.agent_id, self.k, *self.z, *self.s, *self.v]
            msg.data = [float(val) for val in raw_data]
            self.publisher.publish(msg)
            self.k += 1

        else: # Check if we have received messages from all neighbors for the previous iteration
            if all(len(self.received_data[j]) > 0 for j in self.neighbors):
                if all(self.k - 1 == int(self.received_data[j][0][0]) for j in self.neighbors):

                    s_new = self.self_weight * self.s
                    v_new = self.self_weight * self.v
                    
                    for j in self.neighbors:
                        msg_j = self.received_data[j].pop(0) 
                        s_new += self.weights[j] * np.array([msg_j[3], msg_j[4]])
                        v_new += self.weights[j] * np.array([msg_j[5], msg_j[6]])

                    # Calculate Nominal Control
                    g1 = grad1_li(self.z, self.s, self.r, self.gamma, self.beta)
                    g_phi = grad_phi(self.z)
                    u_nom = -self.stepsize * (g1 + g_phi * v_new)

                    # Apply CBF-QP Safety Filter
                    u_app = cbf_qp_filter(self.z, u_nom, self.obstacles, self.d_safe, self.gamma_cbf)

                    z_new = self.z + u_app
                    s_new += phi(z_new) - phi(self.z)

                    grad2_new = grad2_li(z_new, s_new, self.beta)
                    grad2_old = grad2_li(self.z, self.s, self.beta)
                    v_new += grad2_new - grad2_old

                    self.z, self.s, self.v = z_new, s_new, v_new
                    
                    raw_data = [self.agent_id, self.k, *self.z, *self.s, *self.v]
                    msg.data = [float(val) for val in raw_data]
                    self.publisher.publish(msg)

                    self.k += 1
                    if self.k > self.maxK: #Stop after maxK iterations 
                        if self.k == self.maxK + 1:
                            self.get_logger().info("Max iters reached.")
                            self.k += 1 # Ensure this message is only printed once
                        return

def main(args=None):
    rclpy.init(args=args)
    agent = Agent()
    sleep(3.0)
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass # Catch the Ctrl+C silently
    finally:
        agent.destroy_node() 
        rclpy.shutdown()

if __name__ == "__main__":
    main()