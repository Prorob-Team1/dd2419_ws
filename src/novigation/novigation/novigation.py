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


class Navigator(Node):

    def __init__(self):
        super().__init__("navigation")

      
        self.lookahead_distance = 0.5
        self.target_speed = 0.3
        self.goal_tolerance = 0.05
        self.max_off_path_distance = 0.5

       
        self.wheel_base = 0.3125
        self.wheel_radius = 0.04921
        self.max_v = 0.5
        self.max_w = 2 * math.pi / 5

      
        self.path = None
        self.path_idx = 0  # Current progress along path
        self.tail = None
        self.aligning = False
        self._tail_mode = False
        self._aligning_tail = False
        self._last_tail = None
        self._backup_tail = None
        self._backup_idx = 0
        self._object_candidates = []


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.create_subscription(Path, '/planned_path', self.path_callback, 10)
        self.create_subscription(Path, '/tail_path', self.tail_callback, 10)
        self.create_subscription(Empty, '/cancel_navigation', self.cancel_callback, 10)
        self.create_subscription(ObjectCandidateArrayMsg, '/object_candidates', self.candidates_callback, 10)

       
        self.motor_pub = self.create_publisher(
            DutyCycles, "/phidgets/motor/duty_cycles", 10
        )

      
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pure pursuit navigator initialized")

    def candidates_callback(self, msg: ObjectCandidateArrayMsg):
        self._object_candidates = msg.candidates

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            self.get_logger().warn("Received path with fewer than 2 poses, ignoring")
            return

        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path_idx = 0
        self.aligning = True
        self._tail_mode = False

        if self._near_object_candidate(radius=0.6) and self._last_tail is not None:
            self._backup_tail = list(reversed(self._last_tail))
            self._backup_idx = 0
            self.get_logger().info('Object candidate nearby, backing up along reversed tail')

        self.tail = None

        self.get_logger().info(f"Received new path with {len(self.path)} waypoints")

    def tail_callback(self, msg: Path):
        if len(msg.poses) >= 2:
            self.tail = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            self._last_tail = self.tail
        else:
            self.tail = None

    def cancel_callback(self, _msg: Empty):
        self.get_logger().info("Navigation cancelled")
        self.path = None
        self.tail = None
        self._tail_mode = False
        self._backup_tail = None
        self.control_wheels(0.0, 0.0)

    def _near_object_candidate(self, radius=0.6):
        if not self._object_candidates:
            return False
        pose = self.get_robot_pose()
        if pose is None:
            return False
        rx, ry, _ = pose
        for candidate in self._object_candidates:
            dx = candidate.pose.position.x - rx
            dy = candidate.pose.position.y - ry
            if math.hypot(dx, dy) < radius:
                return True
        return False

    def get_robot_pose(self):
        """Get robot (x, y, theta) from TF."""
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

    def control_loop(self):
        if self._backup_tail is None and self.path is None:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        rx, ry, rtheta = pose

        if self._backup_tail is not None:
            tx, ty = self._backup_tail[self._backup_idx]
            dist = math.hypot(tx - rx, ty - ry)
            if dist < self.goal_tolerance:
                self._backup_idx += 1
                if self._backup_idx >= len(self._backup_tail):
                    self._backup_tail = None
                    self.get_logger().info('Backup complete, starting path following')
                    return
                tx, ty = self._backup_tail[self._backup_idx]
            angle_to_target = math.atan2(ty - ry, tx - rx)
            alpha = angle_to_target - (rtheta + math.pi)
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
            w = 2.0 * alpha
            w = max(-self.max_w, min(w, self.max_w))
            v = -0.15
            self.get_logger().info(
                f'BACKUP idx={self._backup_idx}/{len(self._backup_tail)} dist={dist:.2f} alpha={math.degrees(alpha):.1f}° v={v:.2f} w={w:.2f}',
                throttle_duration_sec=0.5
            )
            self.control_wheels(v, w)
            return

        if self.path is None:
            return
        path = self.path

        # Initial alignmenn
        if self.aligning:
            #advances lookahead during aligning phase.
            while self.path_idx < len(path) - 1:
                px, py = path[self.path_idx + 1]
                if math.hypot(px - rx, py - ry) < math.hypot(
                    path[self.path_idx][0] - rx, path[self.path_idx][1] - ry
                ):
                    self.path_idx += 1
                else:
                    break
            look_idx = min(len(path) - 1, self.path_idx + 5)
            target_angle = math.atan2(
                path[look_idx][1] - ry, path[look_idx][0] - rx
            )
            heading_err = target_angle - rtheta
            heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi

            if abs(heading_err) < math.radians(5):
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

        # Advance path_idx
        while self.path_idx < len(path) - 1:
            px, py = path[self.path_idx + 1]
            if math.hypot(px - rx, py - ry) < math.hypot(
                path[self.path_idx][0] - rx, path[self.path_idx][1] - ry
            ):
                self.path_idx += 1
            else:
                break

        # Check if reached final goal (tail tip if tail exists, otherwise path end)
        final_goal = self.tail[-1] if self.tail is not None else path[-1]
        goal_x, goal_y = final_goal
        dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)

        if dist_to_goal < self.goal_tolerance:
            self.get_logger().info("Reached goal, stopping")
            self.control_wheels(0.0, 0.0)
            self.path = None
            self.tail = None
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

        # Go tail mode once we reach the end of the A* path
        if not self._tail_mode and self.tail is not None:
            at_path_end = self.path_idx >= len(path) - 1
            if at_path_end:
                self._tail_mode = True
                self._aligning_tail = True
                self.get_logger().info('Switching to tail/parking mode')

        if self._tail_mode and self.tail is not None:
            tail = self.tail
            tail_heading = math.atan2(tail[-1][1] - tail[0][1], tail[-1][0] - tail[0][0])
            heading_err = tail_heading - rtheta
            heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
            tail_goal_x, tail_goal_y = tail[-1]
            tail_dist = math.hypot(tail_goal_x - rx, tail_goal_y - ry)

            if self._aligning_tail and abs(heading_err) < math.radians(10):
                self._aligning_tail = False
            elif not self._aligning_tail and abs(heading_err) > math.radians(20):
                self._aligning_tail = True

            if self._aligning_tail:
                w = 3.0 * heading_err
                w = max(-self.max_w, min(w, self.max_w))
                self.get_logger().info(
                    f"ALIGN_TAIL err={math.degrees(heading_err):.1f}° w={w:.2f}",
                    throttle_duration_sec=0.5
                )
                self.control_wheels(0.0, w)
            else:
                alpha_p = math.atan2(tail_goal_y - ry, tail_goal_x - rx) - rtheta
                alpha_p = (alpha_p + math.pi) % (2 * math.pi) - math.pi
                eta = heading_err
                beta = eta - alpha_p
                beta = (beta + math.pi) % (2 * math.pi) - math.pi
                v = min(1.2 * tail_dist, 0.2)
                w = 2.5 * alpha_p + 1.0 * beta
                w = max(-self.max_w, min(w, self.max_w))
                self.get_logger().info(
                    f"PARK alpha_p={math.degrees(alpha_p):.1f}° eta={math.degrees(eta):.1f}° "
                    f"v={v:.2f} w={w:.2f} tail_dist={tail_dist:.2f}",
                    throttle_duration_sec=0.5
                )
                self.control_wheels(v, w)
            return

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

        # slow down near transition/goal
        speed = self.target_speed
        slowdown_dist = 0.5
        if self.tail is not None:
            dist_to_transition = math.hypot(path[-1][0] - rx, path[-1][1] - ry)
            if dist_to_transition < slowdown_dist:
                speed = max(0.2, self.target_speed * (dist_to_transition / slowdown_dist))
        elif dist_to_goal < slowdown_dist:
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