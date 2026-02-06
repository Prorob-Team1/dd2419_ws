from sensor_msgs.msg import Joy
import math

import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import  ArmControl, ArmFeedback
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
import time


class RC(Node):

    def __init__(self):
        super().__init__('rc_controller')
      
        self.create_subscription(
            Joy, '/joy', self.joy_callback, 10)
        self.feedback_sub = self.create_subscription(ArmFeedback, "/arm/feedback", self.feedback_callback, 10)
        
        self.arm_pub = self.create_publisher(ArmControl, "/arm/control", 10)
        self.reset_pub = self.create_publisher(Empty, "/arm/reset", 10)
        
        self.get_logger().info("RC node initialized")

        self.cur_pos = [0, 0, 0, 0, 0, 0]
        

    def joy_callback(self, msg: Joy):
        if msg.buttons[6] == 1:
            self.reset_pub.publish(Empty()) # reset
            return


        left_ax = msg.axes[1]
        move = self.cur_pos
        move[2] += 5* left_ax

        time = [300, 300, 300, 300, 300, 300]
        arm_msg = ArmControl()
        arm_msg.position = move
        arm_msg.time = time


        self.get_logger().info(f"{arm_msg=}, {move=}, {left_ax=}")
        self.arm_pub.publish(arm_msg)

        self.get_logger().info(f"{move=}, {left_ax=}")

    def feedback_callback(self, msg: ArmFeedback):
        # self.get_logger().info(f"{msg=}")
        # keep curr pos and set next pos
        self.cur_pos = msg.position


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
    

