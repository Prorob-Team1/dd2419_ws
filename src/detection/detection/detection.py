#!/usr/bin/env python

import math
import numpy as np
import struct
import ctypes

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

# TF2 imports for coordinate transformation
# from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from rclpy.executors import MultiThreadedExecutor
from robp_interfaces.msg import ObjectDetectionMsg, ObjectCandidateArrayMsg
from napping.mapping import ObjectClassification
from typing import Optional
from geometry_msgs.msg import PoseStamped


class Detection(Node):

    def __init__(self):
        super().__init__("detection")

        # TF Buffer to listen for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.object_detection_pub = self.create_publisher(
            ObjectCandidateArrayMsg, "/object_detections", 10
        )

        # Subscribe to point cloud
        self.create_subscription(
            PointCloud2, "/realsense/depth/color/points", self.cloud_callback, 10
        )

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
        points = gen[:, :3]
        colors = np.empty(points.shape, dtype=np.uint32)

        for idx, x in enumerate(gen):
            c = x[3]
            s = struct.pack(">f", c)
            i = struct.unpack(">l", s)[0]
            pack = ctypes.c_uint32(i).value
            colors[idx, 0] = np.asarray((pack >> 16) & 255, dtype=np.uint8)
            colors[idx, 1] = np.asarray((pack >> 8) & 255, dtype=np.uint8)
            colors[idx, 2] = np.asarray(pack & 255, dtype=np.uint8)

        colors = colors.astype(np.float32) / 255

        distance_limit = 1
        distances = np.linalg.norm(points, axis=1)
        close_points = points[distances <= distance_limit]
        close_colors = colors[distances <= distance_limit]

        # color picked and hand tuned, using hsv would be better
        red_mask = (
            (close_colors[:, 0] >= 200 / 255)
            & (close_colors[:, 1] <= 90 / 255)
            & (close_colors[:, 2] <= 90 / 255)
        )
        green_mask = (
            (close_colors[:, 0] <= 20 / 255)
            & (close_colors[:, 1] >= 90 / 255)
            & (close_colors[:, 1] <= 130 / 255)
            & (close_colors[:, 2] >= 80 / 255)
            & (close_colors[:, 2] <= 120 / 255)
        )

        red_points = close_points[red_mask]
        green_points = close_points[green_mask]
        red_colors = close_colors[red_mask]
        green_colors = close_colors[green_mask]

        # filtered_points = np.vstack([red_points, green_points])
        # filtered_colors = np.vstack([red_colors, green_colors])
        detection_msgs = []

        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        object_detection_th = 20
        has_detected_red_object = len(red_points) > object_detection_th
        has_detected_green_object = len(green_points) > object_detection_th
        if has_detected_red_object:
            self.get_logger().info(
                f"{RED}Red Ball spotted ({len(red_points)} points){RESET}"
            )
            detection_msgs.append(
                self.process_object(
                    red_points, ObjectClassification.CUBE_RED, msg.header
                )
            )
        if has_detected_green_object:
            self.get_logger().info(
                f"{GREEN}Green Cube spotted ({len(green_points)} points){RESET}"
            )
            detection_msgs.append(
                self.process_object(
                    green_points, ObjectClassification.CUBE_GREEN, msg.header
                )
            )
        # remove None values from failed detections
        detection_msgs = [d for d in detection_msgs if d is not None]
        self.object_detection_pub.publish(
            ObjectCandidateArrayMsg(detections=detection_msgs)
        )

    def process_object(
        self, points, classification: ObjectClassification, header
    ) -> Optional[ObjectDetectionMsg]:

        centroid = np.mean(points, axis=0)

        p = PointStamped()
        p.header = header
        p.point.x = float(centroid[0])
        p.point.y = float(centroid[1])
        p.point.z = float(centroid[2])

        try:

            trans = self.tf_buffer.lookup_transform(
                "map", header.frame_id, header.stamp, timeout=Duration(seconds=0.1)
            )
            p_map = tf2_geometry_msgs.do_transform_point(p, trans)
            pose_map = PoseStamped()
            pose_map.header = p_map.header
            pose_map.pose.position = p_map.point
            pose_map.pose.orientation.w = 1.0
            detection_msg = ObjectDetectionMsg()
            detection_msg.header.stamp = p_map.header.stamp
            detection_msg.header.frame_id = "map"
            detection_msg.class_name = classification.value
            detection_msg.pose = pose_map.pose
            detection_msg.confidence = 1.0
            return detection_msg

        except Exception as e:
            print("Could not transform")
            return None


def main():
    rclpy.init()
    node = Detection()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
