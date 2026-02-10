from sensor_msgs.msg import Joy
import math

import numpy as np

import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders, DutyCycles
from robp_interfaces.action import ArmExecute
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time


class RC(Node):

    def __init__(self):
        super().__init__('rc_controller')
      
        self.create_subscription(
            Joy, '/joy', self.joy_callback, 10)
        
        self.drive_pub = self.create_publisher(DutyCycles, "/phidgets/motor/duty_cycles", 10)
        
        self.get_logger().info("RC node initialized")

        self.was_pressed = [0] * 11

        self.arm_client = rclpy.action.ActionClient(self, ArmExecute, "arm_execute")

        self.activate_driving = False
        

    def joy_callback(self, msg: Joy):
        left_ax = msg.axes[1]
        right_ax = msg.axes[4]
        drive_msg = DutyCycles()
        # self.get_logger().info(f"{left_ax=}, {right_ax=}")
        drive_msg.duty_cycle_left = left_ax
        drive_msg.duty_cycle_right = right_ax
        drive_msg.header.stamp = msg.header.stamp
        if self.activate_driving:
            self.drive_pub.publish(drive_msg)
        self.handle_button_presses(list(msg.buttons))

    def handle_button_presses(self, buttons: list):
        if buttons[0]: # A
            if not self.was_pressed[0]:
                # Pick & Lift
                command = ArmExecute.Goal()
                command.command = "pick" 
                self.arm_client.send_goal_async(command)
        if buttons[1]: # B
            if not self.was_pressed[1]:
                command = ArmExecute.Goal()
                command.command = "drop" 
                self.arm_client.send_goal_async(command)
        if buttons[2]: # X
            if not self.was_pressed[2]:
                command = ArmExecute.Goal()
                command.command = "lift&pick" 
                self.arm_client.send_goal_async(command)
        if buttons[3]: # Y
            if not self.was_pressed[3]:
                command = ArmExecute.Goal()
                command.command = "lift"
                self.arm_client.send_goal_async(command)

        self.was_pressed = buttons

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
    

