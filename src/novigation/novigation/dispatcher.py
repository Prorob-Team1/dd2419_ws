import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from robp_interfaces.action import Navigation
from geometry_msgs.msg import PoseStamped

class Dispatcher(Node):

    def __init__(self):
        super().__init__('dispatcher_action_client')
        self._action_client = ActionClient(self, Navigation, 'point_navigation')
        self.get_logger().info("Starting dispatcher")

    def send_goal(self, order):
        goal_pose = PoseStamped()
        goal_pose.pose.position.x = 0.
        goal_pose.pose.position.y = 0.
        goal_msg = Navigation.Goal()
        goal_msg.goal = goal_pose

        self._action_client.wait_for_server()

        return self._action_client.send_goal_async(goal_msg)


def main(args=None):
    rclpy.init(args=args)

    action_client = Dispatcher()

    future = action_client.send_goal(10)

    rclpy.spin_until_future_complete(action_client, future)


if __name__ == '__main__':
    main()