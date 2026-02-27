#!/usr/bin/env python

import math

import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from rclpy.time import Time

from collections import deque
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class Odometry(Node):

    def __init__(self):
        super().__init__("odometry")

        # Initialize the transform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Initialize the path publisher
        self._path_pub = self.create_publisher(Path, "path", 10)
        # Store the path here
        self._path = Path()

        # Subscribe to encoder topic and call callback function on each recieved message
        self.create_subscription(
            Encoders, "/phidgets/motor/encoders", self.encoder_callback, 10
        )
        self.create_subscription(
            Imu, "/phidgets/imu/data_raw", self.imu_callback, 10, callback_group=MutuallyExclusiveCallbackGroup()
        )

        # 2D pose
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

        self.last_encoder_left = None
        self.last_encoder_right = None

        # imu psoe
        self._yaw_imu = 0.0
        self._last_imu_stamp = None
        self.w_z_bias = 3e-6 # ! subject to change, likely not a static bias
        self._yaw_lock = threading.Lock()
        self._yaw_imu_queue = deque()

    def imu_callback(self, msg: Imu):
        if self._last_imu_stamp is None:
            self._last_imu_stamp = msg.header.stamp
            return
        w_z = -(msg.angular_velocity.z - self.w_z_bias)
        
        # integrate angular velocity to get yaw
        dt = Time.from_msg(msg.header.stamp) - Time.from_msg(self._last_imu_stamp)
        yaw = self._yaw_imu + w_z * dt.nanoseconds / 1e9
        # wrap angle to [-pi, pi]
        self._yaw_imu = (yaw + np.pi)%(2.0 * np.pi) - np.pi # np.arctan2(np.sin(yaw), np.cos(yaw))
        self._last_imu_stamp = msg.header.stamp

        with self._yaw_lock:
            self._yaw_imu_queue.append((Time.from_msg(msg.header.stamp).nanoseconds, self._yaw_imu))
            if len(self._yaw_imu_queue) > 20: # is this queue size enough?
                self._yaw_imu_queue.popleft()

    def get_yaw(self, encoder_msg_stamp):
        with self._yaw_lock:
            yaw_queue = list(self._yaw_imu_queue) # 
        if not yaw_queue:
            self.get_logger().warning("No gyro readings available, returning last known yaw")
            return self._yaw_imu # return the latest reading
        t_enc = Time.from_msg(encoder_msg_stamp).nanoseconds

        min_t_diff = np.inf
        closest_reading = None
        for i in range(len(yaw_queue)):
            t_imu = yaw_queue[i][0]
            t_diff = abs(t_imu - t_enc)
            if t_diff < min_t_diff:
                closest_reading = yaw_queue[i][1]
                min_t_diff = t_diff
        
        return closest_reading

    def encoder_delta(self, msg: Encoders) -> tuple[int, int]:
        if self.last_encoder_left is None or self.last_encoder_right is None:
            delta_left, delta_right = msg.delta_encoder_left, msg.delta_encoder_right
        else:
            delta_left = msg.encoder_left - self.last_encoder_left
            delta_right = msg.encoder_right - self.last_encoder_right
        self.last_encoder_left = msg.encoder_left
        self.last_encoder_right = msg.encoder_right
        return delta_left, delta_right

    def encoder_callback(self, msg: Encoders):
        """Takes encoder readings and updates the odometry.

        This function is called every time the encoders are updated (i.e., when a message is published on the '/motor/encoders' topic).

        Your task is to update the odometry based on the encoder data in 'msg'. You are allowed to add/change things outside this function.

        Keyword arguments:
        msg -- An encoders ROS message. To see more information about it
        run 'ros2 interface show robp_interfaces/msg/Encoders' in a terminal.
        """

        # The kinematic parameters for the differential configuration
        # dt = 50 / 1000
        ticks_per_rev = 48 * 64
        wheel_radius = 0.04921  # TODO: Fill in
        # base = 0.3135 # TODO: Fill in

        # Ticks since last message
        #delta_ticks_left = msg.delta_encoder_left
        #delta_ticks_right = msg.delta_encoder_right
        delta_ticks_left, delta_ticks_right = self.encoder_delta(msg)

        # TODO: Fill in
        delta_phi_r = 2 * np.pi * (delta_ticks_right / ticks_per_rev)
        delta_phi_l = 2 * np.pi * (delta_ticks_left / ticks_per_rev)

        D = 0.5 * wheel_radius * (delta_phi_r + delta_phi_l)
        # delta_theta = wheel_radius * (delta_phi_r - delta_phi_l) / base

        self._x = self._x + D * np.cos(self._yaw)  # TODO: Fill in
        self._y = self._y + D * np.sin(self._yaw)  # TODO: Fill in
        #self._yaw = self._yaw + delta_theta  # TODO: Fill in
        #self._yaw = np.arctan2(np.sin(self._yaw), np.cos(self._yaw))  # wrap angle
        self._yaw = self.get_yaw(msg.header.stamp)

        stamp = msg.header.stamp  # TODO: Fill in

        self.broadcast_transform(stamp, self._x, self._y, self._yaw)
        self.publish_path(stamp, self._x, self._y, self._yaw)

    def broadcast_transform(self, stamp, x, y, yaw):
        """Takes a 2D pose and broadcasts it as a ROS transform.

        Broadcasts a 3D transform with z, roll, and pitch all zero.
        The transform is stamped with the current time and is between the frames 'odom' -> 'base_link'.

        Keyword arguments:
        stamp -- timestamp of the transform
        x -- x coordinate of the 2D pose
        y -- y coordinate of the 2D pose
        yaw -- yaw of the 2D pose (in radians)
        """

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        # The robot only exists in 2D, thus we set x and y translation
        # coordinates and set the z coordinate to 0
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        # For the same reason, the robot can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        q = quaternion_from_euler(0.0, 0.0, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the transformation
        self._tf_broadcaster.sendTransform(t)

    def publish_path(self, stamp, x, y, yaw):
        """Takes a 2D pose appends it to the path and publishes the whole path.

        Keyword arguments:
        stamp -- timestamp of the transform
        x -- x coordinate of the 2D pose
        y -- y coordinate of the 2D pose
        yaw -- yaw of the 2D pose (in radians)
        """

        self._path.header.stamp = stamp
        self._path.header.frame_id = "odom"

        pose = PoseStamped()
        pose.header = self._path.header

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.01  # 1 cm up so it will be above ground level

        q = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self._path.poses.append(pose)

        self._path_pub.publish(self._path)


def main():
    rclpy.init()
    node = Odometry()
    exectutor = MultiThreadedExecutor()
    exectutor.add_node(node)
    try:
        exectutor.spin()
        #rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    exectutor.shutdown()
    #rclpy.shutdown()


if __name__ == "__main__":
    main()
