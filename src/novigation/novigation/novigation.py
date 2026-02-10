#!/usr/bin/env python

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

# from robp_interfaces.actions import Navigation
from robp_interfaces.msg import Encoders, DutyCycles
from robp_interfaces.action import Navigation
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.action.server import ActionServer, ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor


class Navigator(Node):

    def __init__(self):
        super().__init__("navigation")

        self.k_distance = 0.5
        self.k_heading = 2.0
        self.thresh_distance = 0.1
        self.wheel_base = 0.3
        self.wheel_radius = 0.04921
        self.max_v = 0.5
        self.max_w = 2 * math.pi / 5
        self.time_out = Duration(seconds=10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.nav_action_server = ActionServer(
            self, Navigation, "point_navigation", self.navigate_to_pose_callback
        )
        self.motor_pub = self.create_publisher(
            DutyCycles, "/phidgets/motor/duty_cycles", 10
        )

        self.get_logger().info("Node initialized!")

    def navigate_to_pose_callback(self, goal_handle: ServerGoalHandle):
        rate = self.create_rate(20)
        # save the time when called and use it to timeout if the goal takes too long
        time_start = self.get_clock().now()
        duration_timeout = Duration(seconds=30)

        self.get_logger().info(
            f"Received navigation goal with pose: {goal_handle.request.goal}"
        )

        time_elapsed = self.get_clock().now() - time_start

        while True:
            if time_elapsed > duration_timeout:
                self.get_logger().warn(
                    f"Navigation goal timed out after {duration_timeout} seconds. Stopping robot."
                )
                # stop the robot
                self.control_wheels(0.0, 0.0)
                goal_handle.abort()
                return Navigation.Result(result=False)

            err_distance, err_heading = self.drive_to_goal(goal_handle.request.goal)

            if err_distance < self.thresh_distance:
                self.get_logger().info(
                    f"Reached goal with distance error: {err_distance:.2f}. Stopping robot."
                )
                # stop the robot
                self.control_wheels(0.0, 0.0)
                goal_handle.succeed()
                return Navigation.Result(result=True)

            time_elapsed = self.get_clock().now() - time_start
            rate.sleep()

    def drive_to_goal(self, pose: PoseStamped):
        to_frame_rel = "map"
        from_frame_rel = "base_link"
        try:
            current_pose = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                Time(seconds=0),
                timeout=Duration(seconds=0.1),
            )
            # if the transform is older than 0.1 seconds, consider it invalid
            transform_time = Time.from_msg(current_pose.header.stamp)
            if self.get_clock().now() - transform_time > Duration(
                seconds=0.1
            ):
                raise Exception("Transform is too old")
        except Exception as e:
            self.get_logger().warn(f"Failed to get current pose: {e}")
            return float("inf"), float("inf")

        current_rotation = [
            current_pose.transform.rotation.x,
            current_pose.transform.rotation.y,
            current_pose.transform.rotation.z,
            current_pose.transform.rotation.w,
        ]
        theta_current = euler_from_quaternion(current_rotation)[2]
        dx = pose.pose.position.x - current_pose.transform.translation.x
        dy = pose.pose.position.y - current_pose.transform.translation.y

        err_distance = np.sqrt(dx**2 + dy**2)
        err_heading = math.atan2(dy, dx) - theta_current
        # normalize heading error to [-pi, pi]
        err_heading = (err_heading + math.pi) % (2 * math.pi) - math.pi

        # control
        v = self.k_distance * err_distance
        w = self.k_heading * err_heading

        # spin in place if heading error is too large
        if np.abs(err_heading) > np.pi / 4:
            v = 0

        # clamp velocities
        v = np.clip(v, 0.1, self.max_v)
        w = np.clip(w, -self.max_w, self.max_w)
        self.control_wheels(v, w)

        return err_distance, err_heading

    def control_wheels(self, v: float, w: float):
        # convert to wheel speeds
        left_speed = (v - (w * self.wheel_base / 2)) / self.wheel_radius
        right_speed = (v + (w * self.wheel_base / 2)) / self.wheel_radius

        max_wheel_speed = (
            self.max_v + self.max_w * self.wheel_base / 2
        ) / self.wheel_radius

        # convert range to -1 to 1 for duty cycle
        left = left_speed / max_wheel_speed
        right = right_speed / max_wheel_speed

        m = max(abs(left), abs(right))
        if m > 1.0:
            left /= m
            right /= m

        left_duty_cycle = left * 0.5
        right_duty_cycle = right * 0.5

        self.get_logger().info(
            f"Control command - Left wheel: {left_duty_cycle:.2f}, Right wheel: {right_duty_cycle:.2f}"
        )
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
