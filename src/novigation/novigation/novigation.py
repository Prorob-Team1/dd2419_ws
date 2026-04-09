#!/usr/bin/env python

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion

from robp_interfaces.msg import DutyCycles
from nav_msgs.msg import Path
from std_msgs.msg import Empty, Bool
from geometry_msgs.msg import Point

class Navigator(Node):

    def __init__(self):
        super().__init__("navigation")

        self.lookahead_distance = 0.5
        self.target_speed = 0.3
        self.goal_tolerance = 0.04
        self.max_off_path_distance = 0.5

        self.wheel_base = 0.3135
        self.wheel_radius = 0.04921 - 0.001
        self.max_v = 0.5
        self.max_w = 2 * math.pi / 5

        self.path = None
        self.path_idx = 0
        self.aligning = False
        self._parking_mode = False
        self._aligning_parking = False
        self._parking_enabled = True

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.create_subscription(Path, '/planned_path', self.path_callback, 10)
        self.create_subscription(Bool, '/use_parking', self.parking_callback, 10)
        self.create_subscription(Empty, '/cancel_navigation', self.cancel_callback, 10)
        self.create_subscription(Point, '/move_dist', self.move_dist_callback, 10)

        self.motor_pub = self.create_publisher(
            DutyCycles, "/phidgets/motor/duty_cycles", 10
        )

        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pure pursuit navigator initialized")

    def parking_callback(self, msg: Bool):
        self._parking_enabled = msg.data

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            self.get_logger().warn("Received path with fewer than 2 poses, ignoring")
            return

        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path_idx = 0
        self.aligning = True
        self._parking_mode = False

        self.get_logger().info(f"Received new path with {len(self.path)} waypoints")

    def cancel_callback(self, _msg: Empty):
        self.get_logger().info("Navigation cancelled")
        self.path = None
        self._parking_mode = False
        self._parking_enabled = True
        self.control_wheels(0.0, 0.0)

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link",
                Time(seconds=0),
                timeout=Duration(seconds=0.1),
            )

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

    def move_dist_callback(self, msg: Point):
        self.get_logger().info(f"Received request to move x={msg.x}, y={msg.y}")
        v = 0.2 if msg.x >= 0 else -0.2
        linear_dist = msg.x
        if msg.y != 0 and msg.x >= 0:
            angle_diff = math.atan2(msg.y, msg.x)
            w = 0.1 if angle_diff >= 0 else -0.1
            angular_command_duration = angle_diff / w
            self.control_wheels(0.0, w)
            time.sleep(angular_command_duration)
            linear_dist = math.sqrt(msg.x**2 + msg.y**2)

        linear_command_duration = linear_dist / v
        self.control_wheels(v, 0.0)
        time.sleep(linear_command_duration)
        self.control_wheels(0.0, 0.0)

    def _advance_path_idx(self, path, rx, ry):
        while self.path_idx < len(path) - 1:
            sx, sy = path[self.path_idx]
            ex, ey = path[self.path_idx + 1]
            dx, dy = ex - sx, ey - sy
            t = ((rx - sx) * dx + (ry - sy) * dy) / (dx * dx + dy * dy + 1e-9)
            if t > 0.5:
                self.path_idx += 1
            else:
                break

    def control_loop(self):
        if self.path is None:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        rx, ry, rtheta = pose
        path = self.path

        # Initial alignment
        if self.aligning:
            self._advance_path_idx(path, rx, ry)
            look_idx = min(len(path) - 1, self.path_idx + 5)
            target_angle = math.atan2(
                path[look_idx][1] - ry, path[look_idx][0] - rx
            )
            heading_err = target_angle - rtheta
            heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi

            if abs(heading_err) < math.radians(10):
                self.aligning = False
                self.get_logger().info("Aligned, starting pure pursuit")
            else:
                w = 3.0 * heading_err
                w = max(-self.max_w, min(w, self.max_w))
                self.get_logger().info(
                    f"Aligning: err={math.degrees(heading_err):.1f}° w={w:.2f}",
                    throttle_duration_sec=0.5
                )
                self.control_wheels(0.0, w)
                return

        self._advance_path_idx(path, rx, ry)

        goal_x, goal_y = path[-1]
        dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)

        # Find lookahead point
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

        # Switch to parking mode when close to goal
        if not self._parking_mode and self._parking_enabled:
            if (rx - goal_x)**2 + (ry - goal_y)**2 < 0.5**2:
                self._parking_mode = True
                self._aligning_parking = True
                self.get_logger().info('Switching to parking mode')

        if self._parking_mode:
            heading_to_goal = math.atan2(goal_y - ry, goal_x - rx)
            heading_err = heading_to_goal - rtheta
            heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
            park_dist = math.hypot(goal_x - rx, goal_y - ry)

            if self._aligning_parking and (abs(heading_err) < math.radians(10) or park_dist < 0.08):
                self._aligning_parking = False
            elif not self._aligning_parking and abs(heading_err) > math.radians(20) and park_dist > 0.15:
                

                self._aligning_parking = True

            if self._aligning_parking:
                w = 3.0 * heading_err
                w = max(-self.max_w, min(w, self.max_w))
                self.get_logger().info(
                    f"ALIGN_PARK err={math.degrees(heading_err):.1f}° w={w:.2f}",
                    throttle_duration_sec=0.5
                )
                self.control_wheels(0.0, w)
            else:
                alpha_p = math.atan2(goal_y - ry, goal_x - rx) - rtheta
                alpha_p = (alpha_p + math.pi) % (2 * math.pi) - math.pi
                beta = heading_err - alpha_p
                beta = (beta + math.pi) % (2 * math.pi) - math.pi
                v = max(0.15, min(1.2 * park_dist, 0.2))
                w = 2.5 * alpha_p + 1.0 * beta
                w = max(-self.max_w, min(w, self.max_w))
                self.get_logger().info(
                    f"PARK alpha_p={math.degrees(alpha_p):.1f}° eta={math.degrees(heading_err):.1f}° "
                    f"v={v:.2f} w={w:.2f} dist={park_dist:.2f}",
                    throttle_duration_sec=0.5
                )
                self.control_wheels(v, w)
            return

        # Pure pursuit
        dx = lookahead_pt[0] - rx
        dy = lookahead_pt[1] - ry
        ld = math.hypot(dx, dy)

        alpha = math.atan2(dy, dx) - rtheta
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        speed = self.target_speed
        slowdown_dist = 0.5
        if dist_to_goal < slowdown_dist:
            speed = max(0.25, self.target_speed * (dist_to_goal / slowdown_dist))

        if abs(alpha) > math.pi / 2:
            v = 0.0
            w = 2.0 * alpha
        else:
            ld_for_kappa = max(ld, self.lookahead_distance)
            kappa = 2.0 * math.sin(alpha) / ld_for_kappa
            v = speed
            w = speed * kappa

        v = max(0.0, min(v, self.max_v))
        w = max(-self.max_w, min(w, self.max_w))

        self.get_logger().info(
            f"idx={self.path_idx} alpha={math.degrees(alpha):.1f}° "
            f"theta={math.degrees(rtheta):.1f}° "
            f"v={v:.2f} w={w:.2f} dist_goal={dist_to_goal:.2f}",
            throttle_duration_sec=1.0
        )

        self.control_wheels(v, w)

    def control_wheels(self, v: float, w: float):
        left_speed = (v - (w * self.wheel_base / 2)) / self.wheel_radius
        right_speed = (v + (w * self.wheel_base / 2)) / self.wheel_radius

        max_wheel_speed = (
            self.max_v + self.max_w * self.wheel_base / 2
        ) / self.wheel_radius

        left = left_speed / max_wheel_speed
        right = right_speed / max_wheel_speed

        m = max(abs(left), abs(right))
        if m > 1.0:
            left /= m
            right /= m

        duty_cycles_msg = DutyCycles()
        duty_cycles_msg.duty_cycle_left = left * 0.5
        duty_cycles_msg.duty_cycle_right = right * 0.5
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