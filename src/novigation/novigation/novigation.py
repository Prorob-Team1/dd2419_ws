#!/usr/bin/env python

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from tf2_ros import TransformListener, Buffer
from tf_transformations import euler_from_quaternion

from robp_interfaces.msg import DutyCycles
from nav_msgs.msg import Path
from std_msgs.msg import Empty, Bool
from geometry_msgs.msg import Point, PoseStamped
import numpy as np

class Navigator(Node):

    def __init__(self):
        super().__init__("navigation")

        self.lookahead_distance = 0.5
        self.target_speed = 0.3
        self.goal_tolerance = 0.10
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
        self.parking_goal = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.create_subscription(Path, '/planned_path', self.path_callback, 10)
        self.create_subscription(Bool, '/use_parking', self.parking_callback, 10)
        self.create_subscription(Empty, '/cancel_navigation', self.cancel_callback, 10)
        self.create_subscription(Point, '/move_dist', self.move_dist_callback, 10, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(PoseStamped,'/parking_goal',self.parking_goal_callback,10)
        self.create_subscription(Empty,'/look_around',self.look_around_callback,10)

        self.motor_pub = self.create_publisher(
            DutyCycles, "/phidgets/motor/duty_cycles", 10
        )
        self.look_around_done = self.create_publisher(Empty,"/look_around_done",10)

        self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Pure pursuit navigator initialized")

   


    
    def parking_callback(self, msg: Bool):
        self._parking_enabled = msg.data

    def parking_goal_callback(self, msg: PoseStamped):
        self.parking_goal = (msg.pose.position.x, msg.pose.position.y)

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
        self.parking_goal = None
        self.control_wheels(0.0, 0.0)

    def get_robot_pose(self, from_frame="map"):
        try:
            t = self.tf_buffer.lookup_transform(
                from_frame, "base_link",
                Time(seconds=0),
                timeout=Duration(seconds=0.1),
            )

            transform_time = Time.from_msg(t.header.stamp)
            diff = self.get_clock().now() - transform_time
            if diff > Duration(seconds=0.5):
                self.get_logger().warning(f"Time diff: {diff}")
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
            self.get_logger().info(f"TF lookup failed: {e}")
            return None

    def move_dist_callback(self, msg: Point):
        self.get_logger().info(f"Received request to move x={msg.x}, y={msg.y}")
        start_pose = self.get_robot_pose(from_frame="odom")
        if start_pose is not None:
            s_x, s_y, s_yaw = start_pose
            # this code might fail if the map->odom TF shifts a LOT
            v = 0.2 if msg.x >= 0 else -0.2
            linear_dist = msg.x
            if msg.y != 0:
                # If y is provided, only align
                angle_diff = math.atan2(msg.y, msg.x)
                w = 1.0 if angle_diff >= 0 else -1
                #angular_command_duration = angle_diff / w
                self.control_wheels(0.0, w)
                rotated = 0
                prev_yaw = s_yaw
                start_time = time.time()
                while abs(rotated) < abs(angle_diff):
                    current_pose = self.get_robot_pose(from_frame="odom")
                    if current_pose is not None:
                        c_x, c_y, c_yaw = current_pose
                        # calculate angle diff
                        delta = c_yaw - prev_yaw
                        delta = (delta + np.pi) % (2 * np.pi) - np.pi  # normalize small step

                        rotated += delta
                        prev_yaw = c_yaw
                    else:
                        self.get_logger().warning("Failed to lookup TF :((((((")
                    if time.time() - start_time > 2: # safety first
                        self.get_logger().warning("Rotated for too long, stopping...")
                        break
                    time.sleep(0.02)
                    #self.get_logger().info(f"{rotated=}, {angle_diff=}")
            
    
            else:
                # If y is not provided, DRIVE
                #linear_command_duration = linear_dist / v
                self.control_wheels(v, 0.0)
                traversed_dist = 0
                target_dist = abs(linear_dist)
                start_time = time.time()
                while traversed_dist < target_dist:
                    current_pose = self.get_robot_pose(from_frame="odom")
                    if current_pose is not None:
                        c_x, c_y, c_yaw = current_pose
                        traversed_dist = np.sqrt((c_x - s_x)**2 + (c_y - s_y)**2)
                        self.get_logger().info(f"{traversed_dist=}")
                    else:
                        self.get_logger().warning("Failed to lookup TF :((((((")
                    if time.time() - start_time > 2: # safety first
                        self.get_logger().warning("Drove straight for too long, stopping...")
                        break
                    time.sleep(0.02)
                #time.sleep(linear_command_duration)
            self.control_wheels(0.0, 0.0)

    def look_around_callback(self, msg: Empty):
        self.get_logger().info("Starting look around sequence...")
        start_pose = self.get_robot_pose(from_frame="odom")
        
        if start_pose is None:
            self.get_logger().warn("Failed to get pose, aborting look around.")
            return
            
        s_x, s_y, s_yaw = start_pose
        
       
        targets = [
            s_yaw + math.radians(45),
            s_yaw - math.radians(45),
            s_yaw  
        ]
        
        for target_yaw in targets:
            start_time = time.time()
            
            # Wait and turn until we reach the target
            while rclpy.ok():
                current_pose = self.get_robot_pose(from_frame="odom")
                if current_pose is None:
                    continue
                    
                c_x, c_y, c_yaw = current_pose
                
              
                heading_err = target_yaw - c_yaw
                heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
                
                
                if abs(heading_err) < math.radians(10):
                    self.control_wheels(0.0, 0.0)
                    self.get_logger().info("Reached look target, pausing...")
                    time.sleep(0.5)
                    break
                
               
                w = 3 * heading_err
                w = max(-self.max_w, min(w, self.max_w))
                
                min_abs_w = math.radians(15)  
                if abs(w) < min_abs_w:
                    w = math.copysign(min_abs_w, w)

                self.control_wheels(0.0, w)
                
                if time.time() - start_time > 4.0:
                    self.get_logger().warn("Look around target timed out!")
                    break
                    
                time.sleep(0.02)
        self.look_around_done.publish(Empty())
       
        self.control_wheels(0.0, 0.0)
        self.get_logger().info("Look around sequence complete.")

            
            




    def _advance_path_idx(self, path, rx, ry):
        # Find the closest segment on the remaining path (never go backward).
        # This handles off-path recovery: when the robot drifts, the current
        # segment's projection may never cross 0.5, so we search all remaining
        # segments for the geometrically nearest one.
        best_idx = self.path_idx
        best_dist = float('inf')
        for i in range(self.path_idx, len(path) - 1):
            sx, sy = path[i]
            ex, ey = path[i + 1]
            dx, dy = ex - sx, ey - sy
            seg_len2 = dx * dx + dy * dy + 1e-9
            t = ((rx - sx) * dx + (ry - sy) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
            cx = sx + t * dx
            cy = sy + t * dy
            dist = math.hypot(rx - cx, ry - cy)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        self.path_idx = best_idx

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
                min_abs_w = 3.0 * math.radians(15)
                if abs(w) < min_abs_w: #set lower limit
                    w = math.copysign(min_abs_w,w)


                self.get_logger().info(
                    f"Aligning: err={math.degrees(heading_err):.1f}° w={w:.2f}",
                    throttle_duration_sec=0.5
                )
                self.control_wheels(0.0, w)
                return

        self._advance_path_idx(path, rx, ry)

        goal_x, goal_y = path[-1]
        
        if self.parking_goal is not None:
            parking_goal_x, parking_goal_y = self.parking_goal
        else:
            parking_goal_x, parking_goal_y = goal_x, goal_y
        
        dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)

        #dist_to_parking_goal = math.hypot(parking_goal_x - rx, parking_goal_y)


        #if dist_to_goal < self.goal_tolerance:
            #self.get_logger().info(f"Goal reached (dist={dist_to_goal:.3f}m)")   **THis is probably old artefact, I commented out hope it doesnt break**
            #self.path = None
            #self._parking_mode = False
            #self.control_wheels(0.0, 0.0)
            #return

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
            heading_to_goal = math.atan2(parking_goal_y - ry, parking_goal_x - rx)
            heading_err = heading_to_goal - rtheta
            heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
            park_dist = math.hypot(parking_goal_x - rx, parking_goal_y - ry)

            if self._aligning_parking and (abs(heading_err) < math.radians(10) or park_dist < 0.08):
                self._aligning_parking = False
            elif not self._aligning_parking and abs(heading_err) > math.radians(20) and park_dist > 0.15:
                

                self._aligning_parking = True

            if self._aligning_parking:
                w = 3.0 * heading_err
                w = max(-self.max_w, min(w, self.max_w))
                min_abs_w = 3.0 * math.radians(15)
                if abs(w) < min_abs_w: #set lower limit
                    w = math.copysign(min_abs_w,w)
                self.get_logger().info(
                    f"ALIGN_PARK err={math.degrees(heading_err):.1f}° w={w:.2f}",
                    throttle_duration_sec=0.5
                )
                self.control_wheels(0.0, w)
            else:
                alpha_p = math.atan2(parking_goal_y - ry, parking_goal_x - rx) - rtheta
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
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()