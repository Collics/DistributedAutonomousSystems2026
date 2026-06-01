import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
import numpy as np
from time import sleep

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


class Agent(Node):
    """ROS2 Node representing a single agent in the distributed optimization algorithm."""
    def __init__(self):
        super().__init__(
            "parametric_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        
        # Parameters
        self.stepsize = self.get_parameter("stepsize").value
        self.maxK = self.get_parameter("maxK").value    
        self.gamma = self.get_parameter("gamma").value
        self.beta = self.get_parameter("beta").value
        
        self.r = np.array(self.get_parameter("r").value) # Private target from launch
        
        self.agent_id = float(self.get_parameter("id").value)
        self.neighbors = self.get_parameter("neighbors").value
        
        # self.weights[j] conterrà il peso W_ij
        weight = self.get_parameter("weights").value
        self.weights = {n: w for n, w in zip(self.neighbors, weight)}
        self.self_weight = self.get_parameter("self_weight").value # W_ii

        # Original Log
        self.get_logger().info(f"I am agent: {self.agent_id:.0f}")

        # Initialization of state variables 
        self.k = 0
        self.z = np.array(self.get_parameter("xzero").value)
        self.s = phi(self.z)
        self.v = grad2_li(self.z, self.s, self.beta) # Updated

        # Setup Comunication 
        for j in self.neighbors:
            self.create_subscription(
                MsgFloat,
                f"/topic_{j:.0f}",
                self.listener_callback,
                1000,
            )
        # Buffer to store received messages from neighbors, indexed by neighbor ID
        self.received_data = {j: [] for j in self.neighbors}
        # Publisher for this agent's state
        self.publisher = self.create_publisher(
            MsgFloat,
            f"/topic_{self.agent_id:.0f}",
            1000,
        )

        # Timer (slowed down slightly to 0.05 to allow visualizer to keep up)
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        # Original Log
        self.get_logger().info(f"Agent {self.agent_id:.0f}: setup completed!")

    def listener_callback(self, msg):
        """ When the new msg arrives, move it into the buffer """
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.received_data[j].append(msg_j)

    def timer_callback(self):
        """ When all the msg have arrived, do the update """
        msg = MsgFloat()
        # First iteration: publish initial state without update 
        if self.k == 0:  
            if self.publisher.get_subscription_count() < len(self.neighbors):# Wait until all neighbors are subscribed before publishing
                return
                
            # First Iteration with initial state
            raw_data = [self.agent_id, self.k, *self.z, *self.s, *self.v]
            msg.data = [float(val) for val in raw_data] 
            self.publisher.publish(msg)
            self.k += 1  # Increment iteration counter after publishing initial state

        else:
            # Check if we have received messages from all neighbors for the previous iteration
            all_received = False 
            if all(len(self.received_data[j]) > 0 for j in self.neighbors):
                all_received = all(
                    self.k - 1 == int(self.received_data[j][0][0]) for j in self.neighbors
                )# Ensure that the received messages correspond to the previous iteration (k-1)
                
            if all_received: # If we have all the messages, we can proceed with the update
                s_new = self.self_weight * self.s
                v_new = self.self_weight * self.v
                
                for j in self.neighbors: # Iterate over neighbors to compute the weighted sum of their states
                    msg_j = self.received_data[j].pop(0) 
                    
                    s_j = np.array([msg_j[3], msg_j[4]]) 
                    v_j = np.array([msg_j[5], msg_j[6]])
                    
                    s_new += self.weights[j] * s_j
                    v_new += self.weights[j] * v_j

                g1 = grad1_li(self.z, self.s, self.r, self.gamma, self.beta)
                g_phi = grad_phi(self.z)

                z_new = self.z - self.stepsize * (g1 + g_phi * v_new)

                s_new += phi(z_new) - phi(self.z)

                grad2_new = grad2_li(z_new, s_new, self.beta)
                grad2_old = grad2_li(self.z, self.s, self.beta)

                v_new += grad2_new - grad2_old

                self.z = z_new
                self.s = s_new
                self.v = v_new
                
                # New state
                raw_data = [self.agent_id, self.k, *self.z, *self.s, *self.v]
                msg.data = [float(val) for val in raw_data] 
                self.publisher.publish(msg)

                self.k += 1

                # Stop agent at maxK with original log
                if self.k > self.maxK:
                    if self.k == self.maxK + 1:
                        self.get_logger().info("Max iters reached. Freezing position!")
                        self.k += 1 
                    return

def main(args=None):
    rclpy.init(args=args)
    anAgent = Agent()
    sleep(60)  
    
    try:
        rclpy.spin(anAgent)
    except SystemExit: 
        pass
    finally:
        anAgent.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()