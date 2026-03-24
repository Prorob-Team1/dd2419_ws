#!/usr/bin/env python
"""
Simulated differential drive robot.
Listens to duty cycle commands, integrates kinematics,
broadcasts TF and publishes /current_pose.
"""

import math
import csv

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler

from geometry_msgs.msg import TransformStamped, PoseStamped
from robp_interfaces.msg import DutyCycles


class SimRobot(Node):

    def __init__(self):
        super().__init__('sim_robot')

        # Parameters
        self.declare_parameter('map_file', '/home/robot/Downloads/map_1_1.csv')
        self.declare_parameter('csv_scale', 0.01)
        self.declare_parameter('start_x', 49.0)
        self.declare_parameter('start_y', 50.0)

        # Robot constants (must match navigator)
        self.wheel_base = 0.3125
        self.wheel_radius = 0.04921
        self.max_v = 0.5
        self.max_w = 2 * math.pi / 5
        self.max_wheel_speed = (
            self.max_v + self.max_w * self.wheel_base / 2
        ) / self.wheel_radius

        # Load start position
        csv_scale = self.get_parameter('csv_scale').value
        start_pos = self._load_start(
            self.get_parameter('map_file').value
        )
        if start_pos:
            self.x = start_pos[0] * csv_scale
            self.y = start_pos[1] * csv_scale
        else:
            self.x = self.get_parameter('start_x').value * csv_scale
            self.y = self.get_parameter('start_y').value * csv_scale
        self.theta = 0.0

        # Current duty cycles
        self.duty_left = 0.0
        self.duty_right = 0.0

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Also publish /current_pose for the path planner
        self.pose_pub = self.create_publisher(PoseStamped, '/current_pose', 10)

        # Subscribe to motor commands
        self.create_subscription(
            DutyCycles, '/phidgets/motor/duty_cycles',
            self.duty_callback, 10
        )

        # Simulation loop at 50Hz
        self.dt = 0.02
        self.create_timer(self.dt, self.sim_step)

        self.get_logger().info(
            f'Sim robot started at ({self.x:.2f}, {self.y:.2f})'
        )

    def _load_start(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    if row['Type'].strip() == 'S':
                        return (float(row['x'].strip()), float(row['y'].strip()))
        except Exception:
            pass
        return None

    def duty_callback(self, msg: DutyCycles):
        self.duty_left = msg.duty_cycle_left
        self.duty_right = msg.duty_cycle_right

    def sim_step(self):
        # Reverse the duty cycle → velocity conversion
        left_speed = (self.duty_left / 0.5) * self.max_wheel_speed
        right_speed = (self.duty_right / 0.5) * self.max_wheel_speed

        v_left = left_speed * self.wheel_radius
        v_right = right_speed * self.wheel_radius

        v = (v_left + v_right) / 2.0
        w = (v_right - v_left) / self.wheel_base

        # Integrate
        self.x += v * math.cos(self.theta) * self.dt
        self.y += v * math.sin(self.theta) * self.dt
        self.theta += w * self.dt

        now = self.get_clock().now().to_msg()

        # Broadcast TF: map -> base_link
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, self.theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        # Publish pose for path planner
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = 'map'
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        self.pose_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = SimRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
