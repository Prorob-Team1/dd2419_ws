#!/usr/bin/env python

import math

import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
# from robp_interfaces.actions import Navigation
from robp_interfaces.msg import Encoders
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.action.server import ActionServer, ServerGoalHandle


class Navigation(Node):

    def __init__(self):
        super().__init__("navigation")

        # create tranform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.nav_action_server = ActionServer(
            self, PoseStamped, "navigate_to_pose", self.navigate_to_pose_callback
        )

    def navigate_to_pose_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info(
            f"Received navigation goal with pose: {goal_handle.request.pose}"
        )
        goal_handle.
        pose: PoseStamped = goal_handle.request.pose


def main():
    rclpy.init()
    node = Navigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
