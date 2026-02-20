import py_trees

#!/usr/bin/env python

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

# from robp_interfaces.actions import Navigation
from robp_interfaces.msg import Encoders, DutyCycles
from robp_interfaces.action import Navigation
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.action.server import ActionServer, ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from py_trees.composites import Sequence, Selector, Parallel
from py_trees.behaviour import Behaviour
from py_trees.common import Status


class Brain(Node):

    def __init__(self):
        super().__init__("brain")
        self.tick_period = 0.1

        self.root = py_trees.composites.Sequence("Root", memory=False)
        self.create_timer(self.tick_period, self.root.tick_once)


class GrabCubeB(Behaviour):

    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node

    def update(self):
        self.logger.info("Checking feedback")
        return Status.RUNNING

    def terminate(self, new_status):
        self.logger.info("Interrupted")

    def initialise(self):
        self.logger.info("Call some service")


def main():
    rclpy.init()
    node = Brain()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
