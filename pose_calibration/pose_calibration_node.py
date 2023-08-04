import rclpy
import pandas as pd
import sensor_msgs_py.point_cloud2 as pc
import itertools
import numpy as np 
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.pc_sub = self.create_subscription(PointCloud2, '/camera/depth/color/points', self.pc_cb, 10)

    def pc_cb(self, msg):
        # self.data = msg.data
        cloud = pc.read_points_list(msg, skip_nans=True)

        # apparently faster than converting cloud to np.array directly
        SIZE = len(cloud)
        x = []
        y = []
        z = []
        for idx in range(SIZE):
            if cloud[idx][2] <= 1.5:
                x.append(cloud[idx][0])
                y.append(cloud[idx][1])
                z.append(cloud[idx][2])
        self.cloud_np = np.column_stack((x, y, z))
        self.estimate_pose()

        exit()

    def gram_schmidt(self, vectors):
        num_vectors, vector_length = vectors.shape
        orthogonalized_vectors = np.zeros((num_vectors, vector_length))

        for i in range(num_vectors):
            vector = vectors[i]
            for j in range(i):
                projection = np.dot(vectors[i], orthogonalized_vectors[j]) / np.dot(orthogonalized_vectors[j], orthogonalized_vectors[j])
                vector -= projection * orthogonalized_vectors[j]

            orthogonalized_vectors[i] = vector / np.linalg.norm(vector)

        return orthogonalized_vectors

    def estimate_pose(self):
        
        planes, inliers = self.seq_ransac(self.cloud_np, n_iters=400, threshold=0.005, n_planes=3)

        planes_intersect_vec = []
        for x in itertools.combinations(planes, 2):
            normal1 = x[0][0:3]
            normal2 = x[1][0:3]

            dir_vec = np.cross(normal1, normal2)
            dir_vec_normal = dir_vec / np.linalg.norm(dir_vec)

            z = dir_vec_normal[2]

            # make sure the three vectors are facing the camera
            if z > 0:
                dir_vec_normal = dir_vec_normal * -1
                
            planes_intersect_vec.append(dir_vec_normal)

        # calculating rotation matrix 
        planes_intersect_vec = np.array(planes_intersect_vec)
        orthogonalized_vec = self.gram_schmidt(planes_intersect_vec)
        rotation_mat = np.transpose(orthogonalized_vec)

        # intersection of three planes yields a point
        planes = np.array(planes)
        origin = np.linalg.solve(planes[:, 0:3], -1*planes[:, 3])

        # convert to quaternion
        rotation = Rotation.from_matrix(rotation_mat)
        quaternion = rotation.as_quat()

        print("vectors ", planes_intersect_vec)
        print("orthogonalized ", orthogonalized_vec)
        print("rotation mat ", rotation_mat)
        print("origin ", origin)
        print("quaternion ", quaternion)

    def seq_ransac(self, points, n_iters=1000, threshold=0.005, n_planes=3):
        n_points = len(points)

        points_indices = np.arange(n_points)
        remaining_points = points
            
        final_planes = []
        final_inliers = []
        for i in range(n_planes):

            p = 0.99
            s = 3
            num_inliers = 45000
            eps = (len(remaining_points) - num_inliers) / len(remaining_points)
            N = round(np.log(1-p) / np.log(1-(1-eps)**s))
            print(N)

            # plane, inliers = self.ransac_plane_fit(points=remaining_points, n_iters=n_iters, threshold=threshold)
            plane, inliers = self.ransac_plane_fit(points=remaining_points, n_iters=N, threshold=threshold)

            final_planes.append(plane)
            final_inliers.append(points_indices[inliers])

            # remove the inliers
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            remaining_points = remaining_points[mask]

            # remove the original points indices belong to inliers
            mask2 = np.ones(len(points_indices), dtype=bool)
            mask2[inliers] = False
            points_indices = points_indices[mask2]

        return final_planes, final_inliers

    def ransac_plane_fit(self, points, n_iters=100, threshold=8):

        best_plane_coeffs = None
        best_inliers = None
        max_inliers = 0

        for i in range(n_iters):
            # Randomly select three points to define a plane
            random_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[random_indices]

            # Compute the normal vector of the plane using the selected points
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)

            # Calculate the distance from each point to the plane
            distances = np.abs(np.dot(points - sample_points[0], normal))

            # Count the inliers (points close enough to the plane)
            # inliers = points[distances < threshold]
            inliers = np.where(distances < threshold)[0]
            num_inliers = len(inliers)

            # Update the best plane if necessary
            if num_inliers > max_inliers:
                # Calculate plane coefficients [a, b, c, d]
                d = -np.dot(normal, sample_points[0])
                plane_coeffs = np.append(normal, d)

                best_plane_coeffs = plane_coeffs
                best_inliers = inliers
                max_inliers = num_inliers

        return best_plane_coeffs, best_inliers

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)

