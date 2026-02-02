from sensor_msgs.msg import Joy
import math

import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders, DutyCycles
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time


class RC(Node):

    def __init__(self):
        super().__init__('rc_controller')
      
        self.create_subscription(
            Joy, '/joy', self.joy_callback, 10)
        
        self.drive_pub = self.create_publisher(DutyCycles, "/phidgets/motor/duty_cycles", 10)
        
        

    def joy_callback(self, msg: Joy):
        left_ax = msg.axes[1]
        right_ax = msg.axes[4]
        drive_msg = DutyCycles()
        print(f"{left_ax=}, {right_ax=}")
        drive_msg.duty_cycle_left = left_ax
        drive_msg.duty_cycle_right = right_ax
        drive_msg.header.stamp = msg.header.stamp
        self.drive_pub.publish(drive_msg)

def main():
    rclpy.init()
    node = RC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
    

