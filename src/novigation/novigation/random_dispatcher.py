import rclpy
from rclpy.action.client import ActionClient
from rclpy.node import Node

from robp_interfaces.action import Navigation
from geometry_msgs.msg import PoseStamped
import numpy as np


class RandomDispatcher(Node):

    def __init__(self):
        super().__init__("random_dispatcher")
        self._action_client = ActionClient(self, Navigation, "point_navigation")
        self.square_size = 2.0
        self.min_x = -self.square_size / 2
        self.max_x = self.square_size / 2
        self.min_y = -self.square_size / 2
        self.max_y = self.square_size / 2
        self.nav_goal_future = None

        self.goal_pose_publisher = self.create_publisher(
            PoseStamped, "/dispatcher/goal_pose", 1
        )

        while not self._action_client.wait_for_server(5):
            self.get_logger().error("Navigation action server not ready! Waiting...")

        # ! send first goal, starts endless loop through callback chaining
        self.send_random_goal()

    def send_random_goal(self):
        # sample random x and y coordinates within the square
        x = np.random.uniform(self.min_x, self.max_x)
        y = np.random.uniform(self.min_y, self.max_y)

        goal_pose = PoseStamped()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg = Navigation.Goal()
        goal_msg.goal = goal_pose

        # log the goal being sent
        self.get_logger().info(
            f"Sending navigation goal to pose: ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})"
        )
        # publish the goal pose for visualization
        self.goal_pose_publisher.publish(goal_pose)

        # call action server
        self.nav_goal_future = self._action_client.send_goal_async(goal_msg)
        self.nav_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        nav_goal_handle = future.result()

        if not nav_goal_handle.accepted:
            self.get_logger().error("Navigation goal rejected")
            self.send_random_goal()  # try again
            return

        self.nav_result_future = nav_goal_handle.get_result_async()
        self.nav_result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        try:
            response = future.result()
            result = response.result

            if result.success:
                self.get_logger().info("Navigation goal succeeded!")
            else:
                self.get_logger().error(
                    f"Navigation goal failed with status: {response.status}"
                )

        except Exception as e:
            self.get_logger().error(f"Navigation goal failed: {e}")

        self.send_random_goal()


def main(args=None):
    rclpy.init(args=args)
    action_client = RandomDispatcher()
    rclpy.spin(action_client)
    action_client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
