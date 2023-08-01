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
        gen = pc.read_points_list(msg, skip_nans=True)

        SIZE = len(gen)

        # converting to np array directlly apparently the 2nd slowest in this list
        start_time = time.time()
        gen_np = np.array(gen)
        view = gen_np[:, 0:3]
        print(view)
        # xx = gen_np[:, 0]
        # yy = gen_np[:, 1]
        # zz = gen_np[:, 2]

        print("--- %s seconds ---" % (time.time() - start_time))


        # preallocation of multidimension list -- but it slow :(
        start_time = time.time()

        xyz = [[0 for _ in range(3)] for _ in range(SIZE)]

        for i in range(SIZE):
            for j in range(3):
                xyz[i][j] = gen[i][j]
        final = np.array(xyz)
        print(final)

        print("--- %s seconds ---" % (time.time() - start_time))

        # preallocation executes three times faster than append
        start_time = time.time()

        x = [None] * SIZE
        y = [None] * SIZE
        z = [None] * SIZE
        for idx in range(SIZE):
            x[idx] = gen[idx][0]
            y[idx] = gen[idx][1]
            z[idx] = gen[idx][2]
        xx = np.array(x)
        yy = np.array(y)
        zz = np.array(z)
        stacked = np.column_stack((xx, yy, zz))

        print(stacked)
        print("--- %s seconds ---" % (time.time() - start_time))

        # d = {'x': x,'y': y,'z': z}
        # cloud = PyntCloud(pd.DataFrame(data=d))
        # cloud.to_file("output.ply")

        exit()

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)

