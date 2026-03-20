import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu, CompressedImage
from rclpy.time import Time

from collections import deque
import threading
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from cv_bridge import CvBridge


class ExplorationMapper(Node):
    def __init__(self):
        super().__init__("ExplorationMapper")
        self.get_logger().info("Hello from ExplorationMapper")
        self.bridge = CvBridge()

        self.create_subscription(
            CompressedImage,
            "/realsense/depth/image_rect_raw/compressedDepth",
            self.depth_image_callback,
            1,
        )

    def depth_image_callback(self, msg: CompressedImage):
        self.get_logger().info(
            f"Received compressed depth image with format: {msg.format}"
        )
        try:
            # compressedDepth PNG has a 12-byte header before the actual PNG data.
            # cv_bridge cannot handle this format directly — strip the header manually.
            if "compressedDepth" not in msg.format:
                raise ValueError(f"Unexpected format: {msg.format}")

            # The first 12 bytes are a depth_image_transport header (quantization params).
            # Everything after is a standard PNG that OpenCV can decode.
            raw_data = np.frombuffer(msg.data[12:], dtype=np.uint8)
            image = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError("cv2.imdecode returned None — PNG data may be corrupt")

        except Exception as e:
            self.get_logger().error(f"Conversion failed: {e}")
            return

        self.get_logger().info(
            f"Decoded image shape: {image.shape}, dtype: {image.dtype}"
        )

        # cut off the lower half of the image (the floor)
        height, width = image.shape
        image = image[height // 3 : height // 2, :]

        min_depth = np.min(image)
        max_depth = np.max(image)
        self.get_logger().info(f"Depth range in image: {min_depth} to {max_depth}")

        # mask out invalid depth values (0 or more than 1m)
        depth_display = np.zeros_like(image, dtype=np.uint8)
        valid_mask = (image > 0) & (image < 1000)
        depth_display[valid_mask] = image[valid_mask]

        # show the image in a window (for testing purposes)
        cv2.imshow("Depth Image", depth_display)
        cv2.waitKey(1)  # Needed to update the OpenCV window


def main():
    rclpy.init()
    node = ExplorationMapper()
    exectutor = SingleThreadedExecutor()
    exectutor.add_node(node)
    try:
        exectutor.spin()
        # rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    exectutor.shutdown()
    # rclpy.shutdown()
