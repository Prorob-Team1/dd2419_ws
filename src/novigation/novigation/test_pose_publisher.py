import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import csv


class TestPosePublisher(Node):
    """
    Publishes a static pose for testing path planning.
    Reads start position from map CSV file.
    """

    def __init__(self):
        super().__init__('test_pose_publisher')

        # Parameters
        self.declare_parameter('map_file', '/home/robot/Downloads/map_1_1.csv')
        self.declare_parameter('start_x', 49.0)  # Default from map (in CSV units)
        self.declare_parameter('start_y', 50.0)
        self.declare_parameter('csv_scale', 0.01)  # CSV units to meters (cm -> m)

        map_file = self.get_parameter('map_file').value
        csv_scale = self.get_parameter('csv_scale').value

        # Try to load start position from file
        start_pos = self.load_start_position(map_file)
        if start_pos:
            self.x = start_pos[0] * csv_scale
            self.y = start_pos[1] * csv_scale
        else:
            self.x = self.get_parameter('start_x').value * csv_scale
            self.y = self.get_parameter('start_y').value * csv_scale

        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, '/current_pose', 10)

        # Publish pose at 10Hz
        self.timer = self.create_timer(0.1, self.publish_pose)

        self.get_logger().info(f'Test Pose Publisher initialized at ({self.x:.2f}, {self.y:.2f})')

    def load_start_position(self, filepath):
        """Load start position (Type='S') from map CSV."""
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    if row['Type'].strip() == 'S':
                        x = float(row['x'].strip())
                        y = float(row['y'].strip())
                        self.get_logger().info(f'Found start position: ({x}, {y})')
                        return (x, y)
        except Exception as e:
            self.get_logger().warn(f'Failed to load start position: {e}')
        return None

    def publish_pose(self):
        """Publish current pose."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = 0.0

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
