import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
import numpy as np
from time import sleep
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def local_cost(zi, sigma, r_i, b_i, gamma_i, beta_i):
    return gamma_i * np.dot(zi - r_i, zi - r_i) + beta_i * np.dot(zi - sigma - b_i, zi - sigma - b_i)
 
def grad1_li(zi, sigma, r_i, b_i, gamma_i, beta_i):
    return 2.0 * gamma_i * (zi - r_i) + 2.0 * beta_i * (zi - sigma - b_i)
 
def grad2_li(zi, sigma, b_i, beta_i):
    return -2.0 * beta_i * (zi - sigma - b_i)

def phi(z_i):
    return z_i

def grad_phi(z_i):
    return 1.0


class Agent(Node):
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
        
        self.b = np.array(self.get_parameter("b").value)
        self.r = np.array(self.get_parameter("r").value) # Private target from launch
        
        self.agent_id = float(self.get_parameter("id").value)
        self.neighbors = self.get_parameter("neighbors").value
        
        # self.weights[j] conterrà il peso W_ij
        weight = self.get_parameter("weights").value
        self.weights = {n: w for n, w in zip(self.neighbors, weight)} #couples neighbors and weights in a dict for easier access
        self.self_weight = self.get_parameter("self_weight").value # W_ii

        self.get_logger().info(f"I am agent: {self.agent_id:.0f}")

        # Initialization of state variables 
        self.k = 0
        self.z = np.array(self.get_parameter("xzero").value)
        self.s = phi(self.z)
        self.v = grad2_li(self.z, self.s, self.b, self.beta)

        # Setup Comunication
        for j in self.neighbors:
            self.create_subscription(
                MsgFloat,
                f"/topic_{j:.0f}",
                self.listener_callback,
                10,
            )

        self.received_data = {j: [] for j in self.neighbors}

        self.publisher = self.create_publisher(
            MsgFloat,
            f"/topic_{self.agent_id:.0f}",
            10,
        )

        self.rviz_publisher = self.create_publisher(
            Marker,
            "/agent_markers",
            10
        )

        # Timer 
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.get_logger().info(f"Agent {self.agent_id:.0f}: setup completed!")

    def listener_callback(self, msg):
        """ When the new msg arrives, move it into the buffer """
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.received_data[j].append(msg_j)

    def timer_callback(self):
        """ When all the msg have arrived, do the update """
        msg = MsgFloat()

        if self.k == 0:  
            if self.publisher.get_subscription_count() < len(self.neighbors):
                return
                
            # First Iteration with initial state
            raw_data = [self.agent_id, self.k, *self.z, *self.s, *self.v]
            msg.data = [float(val) for val in raw_data] #ROS2 messages need to be lists of floats, so we convert all values to float before publishing
            self.publisher.publish(msg)
            self.k += 1

        else:
            all_received = False
            # Verifies that there is at least one message from all the neighbors
            if all(len(self.received_data[j]) > 0 for j in self.neighbors):
                # Verifies that the message at the head of the queue is exactly the one from step k-1
                # (msg.data[1] that we have saved in msg_j[0] is the iteration k)
                all_received = all(
                    self.k - 1 == int(self.received_data[j][0][0]) for j in self.neighbors
                )
            if all_received:

                s_new = self.self_weight * self.s
                v_new = self.self_weight * self.v

                
                for j in self.neighbors:
                    
                    msg_j = self.received_data[j].pop(0) 
                    
                    s_j = np.array([msg_j[3], msg_j[4]]) 
                    v_j = np.array([msg_j[5], msg_j[6]])
                    
                    s_new += self.weights[j] * s_j
                    v_new += self.weights[j] * v_j

                g1 = grad1_li(self.z, self.s, self.r, self.b, self.gamma, self.beta)
                g_phi = grad_phi(self.z)

                z_new = self.z - self.stepsize * (g1 + g_phi * v_new)

                s_new += phi(z_new) - phi(self.z)

                grad2_new = grad2_li(z_new, s_new, self.b, self.beta)
                grad2_old = grad2_li(self.z, self.s, self.b, self.beta)

                v_new += grad2_new - grad2_old

                self.z = z_new
                self.s = s_new
                self.v = v_new
                
                # New state
                raw_data = [self.agent_id, self.k, *self.z, *self.s, *self.v]
                msg.data = [float(val) for val in raw_data] #ROS2 messages need to be lists of floats, so we convert all values to float before publishing
                self.publisher.publish(msg)

                self.k += 1

                #Stop agent at maxK
                if self.k > self.maxK:
                    if self.k == self.maxK + 1:
                        self.get_logger().info("Max iters reached. Freezing position!")
                        self.k += 1 # Lo incrementiamo solo per non ripetere il print
                    
                    return

def main(args=None):
    rclpy.init(args=args)
    anAgent = Agent()
    sleep(1)  # Attende che la DDS stabilisca tutte le connessioni prima della prima pubblicazione
    
    try:
        rclpy.spin(anAgent)
    except SystemExit: 
        pass
    finally:
        anAgent.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()