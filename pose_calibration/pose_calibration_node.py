import rclpy
import pandas as pd
import sensor_msgs_py.point_cloud2 as pc
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from pyntcloud import PyntCloud

import ctypes
import struct
import time
import numpy as np 

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.pc_sub = self.create_subscription(PointCloud2, '/camera/depth/color/points', self.pc_cb, 10)

    def pc_cb(self, msg):
        self.data = msg.data
        cloud = pc.read_points_list(msg, skip_nans=True)

        # apparently faster than converting cloud to np.array directly
        start_time = time.time()

        SIZE = len(cloud)

        x = [None] * SIZE
        y = [None] * SIZE
        z = [None] * SIZE
        for idx in range(SIZE):
            x[idx] = cloud[idx][0]
            y[idx] = cloud[idx][1]
            z[idx] = cloud[idx][2]
        cloud_np = np.column_stack((x, y, z))

        print(cloud_np)
        print("--- %s seconds ---" % (time.time() - start_time))

        exit()

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)

