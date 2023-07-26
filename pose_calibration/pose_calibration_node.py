import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import time

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.subscriber1 = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.callback1,
            10  # QoS profile depth
        )

    def callback1(self, msg):
        self.data = msg.data

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)

