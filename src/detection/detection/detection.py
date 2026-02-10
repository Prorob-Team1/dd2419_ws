#!/usr/bin/env python

import math
import numpy as np
import struct
import ctypes

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

# TF2 imports for coordinate transformation
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs 

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

class Detection(Node):

    def __init__(self):
        super().__init__('detection')

        # TF Buffer to listen for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher for Markers (Visual "Placing" of objects)
        self.marker_pub = self.create_publisher(Marker, '/detected_objects', 10)

        # Subscribe to point cloud
        self.create_subscription(
            PointCloud2, '/realsense/depth/color/points', self.cloud_callback, 10)

        self.get_logger().info("Detection node started...")

    def cloud_callback(self, msg: PointCloud2):
        """Takes point cloud readings to detect objects.

        This function is called for every message that is published on the '/camera/depth/color/points' topic.

        Your task is to use the point cloud data in 'msg' to detect objects. You are allowed to add/change things outside this function.

        Keyword arguments:
        msg -- A point cloud ROS message. To see more information about it 
        run 'ros2 interface show sensor_msgs/msg/PointCloud2' in a terminal.
        """

        # Convert ROS -> NumPy
        gen = pc2.read_points_numpy(msg, skip_nans=True)
        
        
        gen = gen[::50]

        if gen.shape[0] == 0:
            return

        points = gen[:, :3] 
      

        rgb_floats = gen[:, 3]
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)

       
        for i, c in enumerate(rgb_floats):
             
             s = struct.pack('>f', c)
             packed = struct.unpack('>l', s)[0]
             
             r = (packed >> 16) & 255
             g = (packed >> 8) & 255
             b = packed & 255
             colors[i] = [r, g, b]

        
        colors /= 255.0

      
        dist_mask = points[:, 2] < 0.9
        
        
        points = points[dist_mask]
        colors = colors[dist_mask]

        if points.shape[0] == 0:
            return


        red_mask = (colors[:, 0] > 0.6) & (colors[:, 1] < 0.4) & (colors[:, 2] < 0.4)
        self.process_object(points, red_mask, "red_cube", [1.0, 0.0, 0.0], msg.header, Marker.CUBE)


        green_mask = (colors[:, 1] > 0.6) & (colors[:, 0] < 0.4) & (colors[:, 2] < 0.4)
        self.process_object(points, green_mask, "green_cube", [0.0, 1.0, 0.0], msg.header, Marker.CUBE)


    def process_object(self, points, mask, name, color_rgb, header, marker_type=Marker.CUBE):

        if np.sum(mask) > 10:

            object_points = points[mask]
            centroid = np.mean(object_points, axis=0)


            p = PointStamped()
            p.header = header
            p.point.x = float(centroid[0])
            p.point.y = float(centroid[1])
            p.point.z = float(centroid[2])


            try:

                trans = self.tf_buffer.lookup_transform('map', header.frame_id, header.stamp)
                p_map = tf2_geometry_msgs.do_transform_point(p, trans)


                self.get_logger().info(f"Found {name} at {p_map.point.x:.2f}, {p_map.point.y:.2f}")


                self.publish_marker(p_map, name, color_rgb, marker_type)

            except TransformException as e:
                print("Could not transform")
                pass

    def publish_marker(self, point_stamped, ns, color, marker_type=Marker.CUBE):
        marker = Marker()
        marker.header = point_stamped.header
        marker.ns = ns
        marker.id = 0
        marker.type = marker_type
        marker.action = Marker.ADD

        marker.pose.position = point_stamped.point
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0

        self.marker_pub.publish(marker)


def main():
    rclpy.init()
    node = Detection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()