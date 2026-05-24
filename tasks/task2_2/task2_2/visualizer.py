import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from visualization_msgs.msg import Marker
import numpy as np


class Visualizer(Node):
    def __init__(self):
        super().__init__(
            'visualizer',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Parametri dal launcher
        self.agent_id = self.get_parameter('agent_id').value
        self.b = np.array(self.get_parameter('b').value)
        self.r = np.array(self.get_parameter('r').value)

        # Stato interno: posizione corrente dell'agente e stima del baricentro
        self.x = 0.0
        self.y = 0.0
        self.sigma = np.zeros(2)

        # Publisher verso RViz (depth 10 per non perdere messaggi)
        self.publisher = self.create_publisher(Marker, '/visualization_topic', 10)

        # Subscription al topic dell'agente corrispondente.
        # I marker vengono pubblicati direttamente nel callback (event-driven)
        # per eliminare il jitter causato da un timer indipendente.
        self.create_subscription(
            MsgFloat,
            f'/topic_{self.agent_id}',
            self.listener_callback,
            10,
        )

    def listener_callback(self, msg):
        # msg.data = [agent_id, k, z0, z1, s0, s1, v0, v1]
        self.x = float(msg.data[2])
        self.y = float(msg.data[3])
        self.sigma = np.array([msg.data[4], msg.data[5]])
        # Pubblica immediatamente: movimento fluido, nessun jitter da timer
        self.publish_data()

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
        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale
        m.color.r = color[0]
        m.color.g = color[1]
        m.color.b = color[2]
        m.color.a = color[3]
        return m

    def publish_data(self):
        # Sfera viola: posizione agente
        self.publisher.publish(self._make_marker(
            marker_id=self.agent_id,
            ns='agents',
            x=self.x,
            y=self.y,
            shape=Marker.SPHERE,
            color=[1.0, 0.0, 1.0, 1.0],
            scale=0.4,
        ))

        # Cubo rosso: target privato r_i (fisso)
        self.publisher.publish(self._make_marker(
            marker_id=self.agent_id + 100,
            ns='targets',
            x=self.r[0],
            y=self.r[1],
            shape=Marker.CUBE,
            color=[1.0, 0.0, 0.0, 1.0],
            scale=0.3,
        ))

        # Cilindro verde: posizione desiderata sigma + b_i
        self.publisher.publish(self._make_marker(
            marker_id=self.agent_id + 200,
            ns='formation',
            x=self.sigma[0] + self.b[0],
            y=self.sigma[1] + self.b[1],
            shape=Marker.CYLINDER,
            color=[0.0, 1.0, 0.0, 1.0],
            scale=0.3,
        ))


def main():
    rclpy.init()
    visualizer = Visualizer()
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("----- Visualizer stopped cleanly -----")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()