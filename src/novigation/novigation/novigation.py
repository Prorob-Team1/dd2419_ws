#!/usr/bin/env python

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion

from robp_interfaces.msg import DutyCycles, ObjectCandidateArrayMsg
from nav_msgs.msg import Path
from std_msgs.msg import Empty

from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion

from robp_interfaces.msg import DutyCycles
from nav_msgs.msg import Path


class Navigator(Node):

    def __init__(self):
        super().__init__("navigation")

        # Pure pursuit parameters
        self.declare_parameter('lookahead_distance', 0.3)
        self.declare_parameter('target_speed', 0.3)
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('max_off_path_distance', 0.5)

        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.target_speed = self.get_parameter('target_speed').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.max_off_path_distance = self.get_parameter('max_off_path_distance').value

        # Robot constants
        self.wheel_base = 0.3
        self.wheel_radius = 0.04921
        self.max_v = 0.5
        self.max_w = 2 * math.pi / 5

        # State
        self.path = None  # List of (x, y) waypoints
        self.path_idx = 0  # Current progress along path (never goes backwards)
        self.aligning = False  # Initial rotation phase before driving

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        # Subscribe to planned path
        self.create_subscription(Path, '/planned_path', self.path_callback, 10)

        # Motor publisher
        self.motor_pub = self.create_publisher(
            DutyCycles, "/phidgets/motor/duty_cycles", 10
        )

        # Control loop at 20Hz
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pure pursuit navigator initialized")

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            self.get_logger().warn("Received path with fewer than 2 poses, ignoring")
            return

        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path_idx = 0
        self.aligning = True
        self.get_logger().info(f"Received new path with {len(self.path)} waypoints")

    def get_robot_pose(self):
        """Get robot (x, y, theta) from TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link",
            t = self.tf_buffer.lookup_transform(
                "map", "base_link",
                Time(seconds=0),
                timeout=Duration(seconds=0.1),
            )
            # Check freshness
            transform_time = Time.from_msg(t.header.stamp)
            if self.get_clock().now() - transform_time > Duration(seconds=0.5):
                return None

            x = t.transform.translation.x
            y = t.transform.translation.y
            q = [
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ]
            theta = euler_from_quaternion(q)[2]
            return (x, y, theta)
        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: {e}")
            return None

    def control_loop(self):
        if self.path is None:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        rx, ry, rtheta = pose
        path = self.path

        # Initial alignment: spin in place to face the path direction
        if self.aligning:
            # Compute direction to a point a bit ahead on the path
            look_idx = min(len(path) - 1, 5)
            target_angle = math.atan2(
                path[look_idx][1] - ry, path[look_idx][0] - rx
            )
            heading_err = target_angle - rtheta
            heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi

            if abs(heading_err) < math.radians(15):
                self.aligning = False
                self.get_logger().info("Aligned, starting pure pursuit")
            else:
                w = 2.0 * heading_err
                w = max(-self.max_w, min(w, self.max_w))
                self.get_logger().info(
                    f"Aligning: err={math.degrees(heading_err):.1f}° w={w:.2f}",
                    throttle_duration_sec=0.5
                )
                self.control_wheels(0.0, w)
                return

        # Advance path_idx forward only
        while self.path_idx < len(path) - 1:
            px, py = path[self.path_idx + 1]
            if math.hypot(px - rx, py - ry) < math.hypot(
                path[self.path_idx][0] - rx, path[self.path_idx][1] - ry
            ):
                self.path_idx += 1
            else:
                break

        # Check if reached goal
        goal_x, goal_y = path[-1]
        dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)

        if dist_to_goal < self.goal_tolerance:
            self.get_logger().info("Reached goal, stopping")
            self.control_wheels(0.0, 0.0)
            self.path = None
            return

        # Find lookahead point: walk along path from path_idx
        lookahead_pt = None
        accumulated = 0.0
        for i in range(self.path_idx, len(path) - 1):
            seg_len = math.hypot(
                path[i + 1][0] - path[i][0],
                path[i + 1][1] - path[i][1]
            )
            if accumulated + seg_len >= self.lookahead_distance:
                remaining = self.lookahead_distance - accumulated
                frac = remaining / seg_len if seg_len > 0 else 0
                lx = path[i][0] + frac * (path[i + 1][0] - path[i][0])
                ly = path[i][1] + frac * (path[i + 1][1] - path[i][1])
                lookahead_pt = (lx, ly)
                break
            accumulated += seg_len

        if lookahead_pt is None:
            lookahead_pt = path[-1]

        # Pure pursuit geometry
        dx = lookahead_pt[0] - rx
        dy = lookahead_pt[1] - ry
        ld = math.hypot(dx, dy)

        if ld < 1e-6:
            self.control_wheels(0.0, 0.0)
            return

        # Angle from robot heading to lookahead point
        alpha = math.atan2(dy, dx) - rtheta
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        # Speed: slow down near goal
        speed = self.target_speed
        slowdown_dist = 0.5
        if dist_to_goal < slowdown_dist:
            speed = max(0.05, self.target_speed * (dist_to_goal / slowdown_dist))

        # Proportional spin when heading is very off, otherwise pure pursuit
        if abs(alpha) > math.pi / 2:
            # Large heading error: spin in place proportionally
            v = 0.0
            w = 2.0 * alpha  # proportional gain
        else:
            # Pure pursuit curvature
            kappa = 2.0 * math.sin(alpha) / ld
            v = speed
            w = speed * kappa

        # Clamp
        v = max(0.0, min(v, self.max_v))
        w = max(-self.max_w, min(w, self.max_w))

        self.get_logger().info(
            f"idx={self.path_idx} alpha={math.degrees(alpha):.1f}° "
            f"v={v:.2f} w={w:.2f} dist_goal={dist_to_goal:.2f}",
            throttle_duration_sec=1.0
        )

        self.control_wheels(v, w)

    def control_wheels(self, v: float, w: float):
        # Convert to wheel speeds
        left_speed = (v - (w * self.wheel_base / 2)) / self.wheel_radius
        right_speed = (v + (w * self.wheel_base / 2)) / self.wheel_radius

        max_wheel_speed = (
            self.max_v + self.max_w * self.wheel_base / 2
        ) / self.wheel_radius

        # Convert range to -1 to 1 for duty cycle
        left = left_speed / max_wheel_speed
        right = right_speed / max_wheel_speed

        m = max(abs(left), abs(right))
        if m > 1.0:
            left /= m
            right /= m

        left_duty_cycle = left * 0.5
        right_duty_cycle = right * 0.5

        duty_cycles_msg = DutyCycles()
        duty_cycles_msg.duty_cycle_left = left_duty_cycle
        duty_cycles_msg.duty_cycle_right = right_duty_cycle
        duty_cycles_msg.header.stamp = self.get_clock().now().to_msg()
        self.motor_pub.publish(duty_cycles_msg)


def main():
    rclpy.init()
    node = Navigator()
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
