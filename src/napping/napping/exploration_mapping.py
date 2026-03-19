import math

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


class ExplorationMapper(Node):
    def __init__(self):
        super().__init__("ExplorationMapper")
        self.get_logger().info("Hello from ExplorationMapper")

        self.create_subscription(CompressedImage, "/realsense/depth/image_rect_raw/compressedDepth", self.depth_image_callback, 1)

    
    def depth_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)

        self.get_logger().info(f"Image shape {np_arr.shape}")

    


def main():
    rclpy.init()
    node = ExplorationMapper()
    exectutor = SingleThreadedExecutor()
    exectutor.add_node(node)
    try:
        exectutor.spin()
        #rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    exectutor.shutdown()
    #rclpy.shutdown()