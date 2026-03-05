#!/usr/bin/env python

import time

import rclpy
from rclpy.action.server import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from robp_interfaces.action import DummyAction


class DummyActionServer(Node):

    def __init__(self):
        super().__init__("dummy_action_server")
        self._action_server = ActionServer(
            self,
            DummyAction,
            "dummy",
            execute_callback=self._execute_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info("Dummy action server started")

    def _execute_callback(self, goal_handle: ServerGoalHandle) -> DummyAction.Result:
        self.get_logger().info(f"Received goal: succeed={goal_handle.request.succeed}")

        # Publish feedback and wait 5 seconds
        for i in range(5):
            feedback = DummyAction.Feedback()
            feedback.status = f"Waiting... {i + 1}/5"
            goal_handle.publish_feedback(feedback)
            time.sleep(1.0)

        if goal_handle.request.succeed:
            result = DummyAction.Result()
            result.success = True
            goal_handle.succeed()
            self.get_logger().info("Goal succeeded")
            return result
        else:
            result = DummyAction.Result()
            result.success = False
            goal_handle.abort()
            self.get_logger().info("Goal aborted (succeed=False)")
            return result


def main():
    rclpy.init()
    node = DummyActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
