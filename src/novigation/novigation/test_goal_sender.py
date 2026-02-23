import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from robp_interfaces.action import Navigation
from geometry_msgs.msg import PoseStamped
import sys


class TestGoalSender(Node):
    """
    Test client that forwards goals to the path planner action server.
    Accepts goals from:
      - RViz2 "2D Goal Pose" button (subscribes to /goal_pose)
      - Command line: ros2 run novigation test_goal_sender <x> <y>
    """

    def __init__(self):
        super().__init__('test_goal_sender')

        self._action_client = ActionClient(
            self,
            Navigation,
            'plan_path'
        )

        # Subscribe to RViz2 goal pose
        self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.rviz_goal_callback,
            10
        )

        self.get_logger().info('Test Goal Sender initialized')
        self.get_logger().info('Click "2D Goal Pose" in RViz2 to send goals')

    def rviz_goal_callback(self, msg: PoseStamped):
        """Handle goal from RViz2."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.get_logger().info(f'Received RViz goal: ({x:.2f}, {y:.2f})')
        self.send_goal(x, y)

    def send_goal(self, x, y):
        """Send a goal to the path planner."""
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Action server not available!')
            return

        goal_msg = Navigation.Goal()
        goal_msg.goal.header.frame_id = 'map'
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = x
        goal_msg.goal.pose.position.y = y
        goal_msg.goal.pose.position.z = 0.0
        goal_msg.goal.pose.orientation.w = 1.0

        self.get_logger().info(f'Sending goal: ({x:.2f}, {y:.2f})')

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback.feedback
        self.get_logger().info(f'Feedback: {feedback}')

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by server!')
            return

        self.get_logger().info('Goal accepted! Waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result

        if result.result:
            self.get_logger().info('Path planning succeeded!')
        else:
            self.get_logger().error('Path planning failed!')


def main(args=None):
    rclpy.init(args=args)
    node = TestGoalSender()

    # If CLI args provided, send that goal too
    if len(sys.argv) >= 3:
        try:
            x = float(sys.argv[1])
            y = float(sys.argv[2])
            node.send_goal(x, y)
        except ValueError:
            node.get_logger().error('Invalid coordinates!')

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
