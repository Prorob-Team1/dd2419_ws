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
from rclpy.action.client import ActionClient
from rclpy.executors import MultiThreadedExecutor
from py_trees.composites import Sequence, Selector, Parallel
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from action_msgs.msg import GoalStatus

from robp_interfaces.action import DummyAction


class Brain(Node):

    def __init__(self):
        super().__init__("brain")
        self.tick_period = 0.1

        # self.root = py_trees.composites.Sequence("Root", memory=False)
        # self.create_timer(self.tick_period, self.root.tick_once)

        self._dummy_action_client = ActionClient(self, DummyAction, "dummy")
        while not self._dummy_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().info("Waiting for dummy action server...")

        self._dummy_action_client.wait_for_server()
        goal = DummyAction.Goal(succeed=True)
        send_goal_future = self._dummy_action_client.send_goal_async(
            goal, feedback_callback=self.dummy_feedback_callback
        )
        send_goal_future.add_done_callback(self.dummy_goal_response_callback)

    def dummy_feedback_callback(self, feedback_msg):
        self.get_logger().info(f"Feedback: {feedback_msg.feedback.status}")

    def dummy_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return
        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.dummy_result_callback)

    def dummy_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result received: success={result}")
        self.get_logger().info(f"Result: success={result.success}")

        try:
            response = future.result()
            result = response.result

            if response.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"Action goal succeeded! {result}")
            else:
                self.get_logger().error(
                    f"Action goal failed with status: {response.status}"
                )

        except Exception as e:
            self.get_logger().error(f"Action goal failed: {e}")


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
