import rclpy
# import pandas as pd
import sensor_msgs_py.point_cloud2 as pc
# import itertools
import numpy as np 
import struct
import ctypes
import cv2
import random
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
# from tf2_msgs.msg import TFMessage
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster

class Plane():
  def __init__(self, three_points):
    self.three_points = three_points
    self.a, self.b, self.c, self.d = self.fitPlane(three_points)
    self.inlier_indices = []
    self.normal = [self.a, self.b, self.c]
    self.coeff = [self.a, self.b, self.c, self.d]

  def fitPlane(self, three_points):
    v1 = three_points[1] - three_points[0]
    v2 = three_points[2] - three_points[0]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    d = -np.dot(normal, three_points[0])
    return normal[0], normal[1], normal[2], d

class Cloud():
  # helper class to deal with ros PointCloud2
  def __init__(self, cloud):
    self.points, self.rgb = self.unpack_to_numpy(cloud)

  def unpack_to_numpy(self, cloud):
    SIZE = len(cloud)
    x=[]; y=[]; z=[]; r=[]; g=[]; b=[]
    for idx in range(SIZE):
      if cloud[idx][2] <= 1.0:
        x.append(cloud[idx][0])
        y.append(cloud[idx][1])
        z.append(cloud[idx][2])

        test = cloud[idx][3]
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f' ,test)
        i = struct.unpack('>l',s)[0]

        pack = ctypes.c_uint32(i).value
        rr = (pack & 0x00FF0000)>> 16
        gg = (pack & 0x0000FF00)>> 8
        bb = (pack & 0x000000FF)

        r.append(rr)
        g.append(gg)
        b.append(bb)

    points = np.column_stack((x, y, z))
    rgb = np.column_stack((r, g, b))
    return points, rgb

class MyNode(Node):
  def __init__(self):
    super().__init__('my_node')
    self.pc_sub = self.create_subscription(PointCloud2, '/camera/depth/color/points', self.pc_cb, 10)
    self.calib_srv = self.create_service(Empty, '/pose_calibration/get_pose', self.calib_cb)
    # self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)  
    self.tf_broadcaster = TransformBroadcaster(self)
    self.timer = self.create_timer(0.1, self.publish_transform)
    self.points = []
    # transformation mat
    self.t_cw = None
    self.t_cr = None
    self.t_ct = None

  def publish_transform(self):
    if self.t_cw is None: return

    def create_transform(node, t, parent, child):
      r = t[:3, :3]
      quaternion = Rotation.from_matrix(r).as_quat()
      translation = t[:3, 3]
      transform_msg = TransformStamped()
      transform_msg.header.stamp = node.get_clock().now().to_msg()
      transform_msg.header.frame_id = parent
      transform_msg.child_frame_id = child
      transform_msg.transform.translation.x = translation[0]
      transform_msg.transform.translation.y = translation[1]
      transform_msg.transform.translation.z = translation[2]
      transform_msg.transform.rotation.x = quaternion[0]
      transform_msg.transform.rotation.y = quaternion[1]
      transform_msg.transform.rotation.z = quaternion[2]
      transform_msg.transform.rotation.w = quaternion[3]
      return transform_msg

    tcw_msg = create_transform(self, self.t_cw, 'camera_depth_optical_frame', 'calibration_box')
    tcr_msg = create_transform(self, self.t_cr, 'camera_depth_optical_frame', 'robot_base')
    tct_msg = create_transform(self, self.t_ct, 'camera_depth_optical_frame', 'tool')

    self.tf_broadcaster.sendTransform(tcw_msg)
    self.tf_broadcaster.sendTransform(tcr_msg)
    # self.tf_broadcaster.sendTransform(tct_msg)

  def pc_cb(self, msg):
    self.points = pc.read_points_list(msg, skip_nans=True)

  def calib_cb(self, request, response):
    if self.points == []: print("Point cloud is empty!"); return response
    print("Request received...")
    cloud = Cloud(self.points)
    planes = seq_ransac(cloud) # run sequential ransac
    # export_planes(cloud.points, planes) 

    # identify the plane
    # need refactor
    color_indices = []
    for i in range(3):
      rgb = cloud.rgb[planes[i].inlier_indices]
      rgb = rgb.astype(np.uint8)
      hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
      hsv_squeezed = np.squeeze(hsv, axis=1)

      red_mask = np.logical_or(hsv_squeezed[:, 0] > 170, hsv_squeezed[:, 0] < 10)
      green_mask = np.logical_and(hsv_squeezed[:, 0] > 45, hsv_squeezed[:, 0] < 75)
      blue_mask = np.logical_and(hsv_squeezed[:, 0] > 90, hsv_squeezed[:, 0] < 110)

      red_pcs = hsv_squeezed[red_mask]
      green_pcs = hsv_squeezed[green_mask]
      blue_pcs = hsv_squeezed[blue_mask]

      print(f"r:{len(red_pcs)} g:{len(green_pcs)} b:{len(blue_pcs)}")

      color_idx = np.argmax([len(red_pcs), len(green_pcs), len(blue_pcs)])
      color_indices.append(color_idx)

    print(f"indices: {color_indices}")
    red_plane = planes[np.where(np.array(color_indices)==0)[0][0]]
    green_plane = planes[np.where(np.array(color_indices)==1)[0][0]]
    blue_plane = planes[np.where(np.array(color_indices)==2)[0][0]]

    x_vec = calc_vec(red_plane, blue_plane)
    y_vec = calc_vec(green_plane, red_plane)
    z_vec = calc_vec(green_plane, blue_plane)
    planes_intersect_vec = np.vstack((x_vec, y_vec, z_vec))

    orthogonalized_vec = gram_schmidt(np.asarray(planes_intersect_vec)) # orthogonalize vectors
    rotation_mat = np.transpose(orthogonalized_vec)
    quaternion = Rotation.from_matrix(rotation_mat).as_quat()

    # find three planes intersection
    origin = np.linalg.solve(np.array([planes[i].normal for i in range(3)]), -1*np.array([planes[i].d for i in range(3)]))

    # find t_ct
    # transformation of world w.r.t camera
    self.t_cw = np.eye(4)
    self.t_cw [:3, :3] = rotation_mat 
    self.t_cw [:3, 3] = origin 

    # transformation of robot's base w.r.t world
    t_rw = np.array([[-1.0000000,  0.0000000,  0.0000000, 0.314],
                     [0.0000000,  -1.0000000,  0.0000000, 0.947],
                     [0.0000000,  0.0000000,  1.0000000, 0.029],
                     [0.0000000, 0.0000000, 0.0000000, 1.0000000]])
    t_wr = np.linalg.inv(t_rw)

    # transformation of arm w.r.t robot's base
    t_rt = np.array([[0.3520344, -0.7203782, -0.5976012, -0.174470],
                     [0.5511497,  0.6755850, -0.4897130, 0.560472],
                     [0.7565089, -0.1569719,  0.6348653, 0.680237],
                     [0.0000000, 0.0000000, 0.0000000, 1.0000000]])
    
    self.t_cr = self.t_cw @ t_wr
    self.t_ct = self.t_cr @ t_rt
    # print(f"t_cw: {t_cw}\n t_rw: {t_wr}\n t_rt: {t_rt}\n")

    # print for debugging 
    print("vectors ", planes_intersect_vec)
    print("orthogonalized ", orthogonalized_vec)
    print("rotation mat ", rotation_mat)
    print("origin ", origin)
    print("quaternion ", quaternion)
    return response


def seq_ransac(cloud, threshold=0.005, n_planes=3):
  def ransac_plane_fit(points, n_iters):
    best_plane = []
    best_inliers = []
    max_inliers = 0

    for i in range(n_iters):
      random_indices = np.random.choice(len(points), 3, replace=False) # select three points randomly
      sample_points = points[random_indices]
      plane = Plane(sample_points) # fit a plane 
      distances = np.abs(np.dot(points - sample_points[0], plane.normal)) # calc distance
      inliers = np.where(distances < threshold)[0]
      num_inliers = len(inliers) # calc num of inliers
      # Update the best plane if necessary
      if num_inliers > max_inliers:
        best_plane = plane
        best_plane.inlier_indices = inliers
        max_inliers = num_inliers
    return best_plane

  n_points = len(cloud.points)
  points_indices = np.arange(n_points)
  remaining_points = cloud.points

  final_planes = []
  final_inliers = []
  for i in range(n_planes):
    # roughly calculate num of iterations (N)
    p = 0.99; s = 3; num_inliers = 45000
    eps = (len(remaining_points) - num_inliers) / len(remaining_points)
    N = round(np.log(1-p) / np.log(1-(1-eps)**s)) 
    N = 1000 if N > 1000 else N
    plane = ransac_plane_fit(remaining_points, N) # run ransac
    # update the inlier to the global indices after removing
    local_points_indices = plane.inlier_indices.copy()
    plane.inlier_indices[:] = points_indices[local_points_indices]
    final_planes.append(plane)
    # remove the inliers
    mask = np.ones(len(remaining_points), dtype=bool)
    mask[local_points_indices] = False
    remaining_points = remaining_points[mask]
    # remove the original points indices belong to inliers
    mask2 = np.ones(len(points_indices), dtype=bool)
    mask2[local_points_indices] = False
    points_indices = points_indices[mask2]
  return final_planes

def gram_schmidt(vectors):
  num_vectors, vector_length = vectors.shape
  orthogonalized_vectors = np.zeros((num_vectors, vector_length))

  for i in range(num_vectors):
    vector = vectors[i]
    for j in range(i):
      projection = np.dot(vectors[i], orthogonalized_vectors[j]) / np.dot(orthogonalized_vectors[j], orthogonalized_vectors[j])
      vector -= projection * orthogonalized_vectors[j]
    orthogonalized_vectors[i] = vector / np.linalg.norm(vector)

  return orthogonalized_vectors

def calc_vec(plane1, plane2):
  dir_vec = np.cross(plane1.normal, plane2.normal)
  dir_vec_normal = dir_vec / np.linalg.norm(dir_vec)

  z = dir_vec_normal[2]

  # make sure the three vectors are facing the camera
  if z > 0: dir_vec_normal = dir_vec_normal * -1

  return dir_vec_normal

def export_planes(points, planes):
  length = len(points)
  rgb = np.full((length, 3), (255, 255, 255)) 

  rgb[planes[0].inlier_indices] = (255, 0, 0)
  rgb[planes[1].inlier_indices] = (0, 255, 0)
  rgb[planes[2].inlier_indices] = (0, 0, 255)

  rgb_np = np.array(rgb)
  ver = np.hstack((points, rgb_np))
  new_ver = np.core.records.fromarrays(ver.transpose(), 
                                       names='x, y, z, red, green, blue',
                                       formats = 'f4, f4, f4, u1, u1, u1')
  el = PlyElement.describe(new_ver, 'vertex')
  PlyData([el], text=True).write('./ascii.ply')

def main(args=None):
  rclpy.init(args=args)
  node = MyNode()
  rclpy.spin(node)
