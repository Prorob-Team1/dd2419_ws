import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robp_interfaces.action import Navigation
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from action_msgs.msg import GoalStatus
import sys


class GoalReceiver(Node):

    def __init__(self):
        super().__init__('goal_receiver')

        self._action_client = ActionClient(self, Navigation, 'plan_path')
        self._goal_handle = None

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(Empty, '/cancel_navigation', self.cancel_callback, 10)

        self.get_logger().info('Goal receiver ready')

    def goal_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.get_logger().info(f'Received goal: ({x:.2f}, {y:.2f})')
        if self._goal_handle is not None:
            self.get_logger().info('Cancelling current goal before sending new one')
            self._goal_handle.cancel_goal_async()
        self.send_goal(x, y)

    def cancel_callback(self, _msg: Empty):
        if self._goal_handle is not None:
            self.get_logger().info('Cancelling navigation goal')
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None

    def send_goal(self, x, y):
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Action server not available')
            return

        goal_msg = Navigation.Goal()
        goal_msg.goal.header.frame_id = 'map'
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = x
        goal_msg.goal.pose.position.y = y
        goal_msg.goal.pose.orientation.w = 1.0

        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return
        self._goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda f, gh=goal_handle: self.result_callback(f, gh)
        )

    def result_callback(self, future, original_handle):
        wrapped = future.result()
        is_current = (self._goal_handle is original_handle)

        if wrapped.status == GoalStatus.STATUS_CANCELED:
            if is_current:
                self._goal_handle = None
                self.get_logger().info('Goal cancelled')
            else:
                self.get_logger().info('Old goal cancelled, new goal active')
            return

        if is_current:
            self._goal_handle = None

        if wrapped.result.result:
            self.get_logger().info('Navigation succeeded')
        else:
            self.get_logger().error('Navigation failed')


def main(args=None):
    rclpy.init(args=args)
    node = GoalReceiver()

    if len(sys.argv) >= 3:
        try:
            node.send_goal(float(sys.argv[1]), float(sys.argv[2]))
        except ValueError:
            node.get_logger().error('Invalid coordinates')

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()