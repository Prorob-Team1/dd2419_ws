#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.action.server import ActionServer, ServerGoalHandle
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from robp_interfaces.action import ArmExecute
from robp_interfaces.msg import ArmControl, ArmFeedback
import numpy as np


DEFAULT_PICK_POSE = [97.0, 120.0, 70.0, 185.0, 65.0, 120.0]
DEFAULT_PICK_TIME_MS = [4000, 2000, 2000, 2000, 2000, 2000]

DEFAULT_DROP_POSE = [97.0, 120.0, 70.0, 185.0, 65.0, 120.0]
DEFAULT_DROP_TIME_MS = [1000, 1000, 1000, 1000, 1000, 1000]

DEFAULT_LIFT_POSE = [97.0, 120.0, 70.0, 105.0, 100.0, 120.0]
DEFAULT_LIFT_TIME_MS = [1000, 1000, 1000, 1000, 1000, 1000]

OPEN_GRIPPER_TIME_MS = 500 # min must be 500ms?
OPEN_GRIPPER_POSITION = [30, 120.0]

POSITION_TOLERANCE = 10.0 # degrees
TIMEOUT_SECONDS = 10


def _to_uint16_list(values):
    """Ensure 6 ints in uint16 range for ArmControl.time."""
    n = list(values)[:6]
    while len(n) < 6:
        n.append(3000)
    return [max(0, min(65535, int(v))) for v in n]


class ArmMoveActionServer(Node):
    def __init__(self):
        super().__init__("arm_move_action_server")
 


        self._current_position = None
        self._feedback_sub = self.create_subscription(
            ArmFeedback, "/arm/feedback", self._feedback_callback, 10
        )
        self._control_pub = self.create_publisher(ArmControl, "/arm/control", 10)

        self._action_server = ActionServer(self, ArmExecute, "arm_execute", self._execute_callback)

        self.get_logger().info(f"ArmMoveActionServer initialized")

    def _feedback_callback(self, msg: ArmFeedback) -> None:
        self._current_position = list(msg.position)

    def _get_pose(self, command: str):
        "Return (function, timeout_seconds)"
        c = command.strip().lower()
        if c == "pick":
            return self._pick
        if c == "drop":
            return self._drop
        if c == "lift":
            return self._lift
        if c == "lift&pick":
            return self._pick_and_lift
        return None

    def _execute_callback(self, goal_handle: ServerGoalHandle) -> ArmExecute.Result:
        command = goal_handle.request.command
        func_to_execute = self._get_pose(command)
        if func_to_execute is None:
            self.get_logger().error("Unknown command '%s'; use pick, drop, or lift." % command)
            goal_handle.abort()
            return ArmExecute.Result(success=False)
        try:
            result = func_to_execute()
        except Exception as e:
            self.get_logger().error("Execute failed: %s" % e)
            goal_handle.abort()
            return ArmExecute.Result(success=False)
        success = bool(result) if result is not None else False
        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return ArmExecute.Result(success=success)

    def _pick(self):
        rate = self.create_rate(10)
        clock = self.get_clock()

        # Optionally open gripper first, then move to pick pose
        if (not np.allclose(self._current_position[0], OPEN_GRIPPER_POSITION[0], atol=POSITION_TOLERANCE)
            or not np.allclose(self._current_position[1], OPEN_GRIPPER_POSITION[1], atol=POSITION_TOLERANCE)):
            target_position = self._current_position.copy()
            target_position[0] = OPEN_GRIPPER_POSITION[0]
            target_position[1] = OPEN_GRIPPER_POSITION[1]

            ctrl = ArmControl()
            ctrl.header.stamp = clock.now().to_msg()
            ctrl.position = target_position
            ctrl.time = _to_uint16_list(OPEN_GRIPPER_TIME_MS * [1, 1, 1, 1, 1, 1])
            self._control_pub.publish(ctrl)
            time_start = clock.now()
            while (clock.now() - time_start) < Duration(seconds=OPEN_GRIPPER_TIME_MS*1.5 / 1000.0 + 1):
                rate.sleep()
            if not np.allclose(self._current_position, target_position, atol=POSITION_TOLERANCE):
                self.get_logger().error("Failed to open gripper")
                self.get_logger().error(f"Current position: {self._current_position}")
                self.get_logger().error(f"Target position: {target_position}")
                return False
                
        # Move to pick pose
        ctrl = ArmControl()
        ctrl.header.stamp = clock.now().to_msg()
        ctrl.position = list(DEFAULT_PICK_POSE)
        ctrl.time = _to_uint16_list(DEFAULT_PICK_TIME_MS)
        self._control_pub.publish(ctrl)
        time_start = clock.now()
        while (clock.now() - time_start) < Duration(seconds=max(DEFAULT_PICK_TIME_MS)*1.5 / 1000.0):
            rate.sleep()

        if np.allclose(DEFAULT_PICK_POSE, self._current_position, atol=POSITION_TOLERANCE):
            return True
        self.get_logger().error("Failed to move to pick pose")
        self.get_logger().error(f"Current position: {self._current_position}")
        self.get_logger().error(f"Target position: {DEFAULT_PICK_POSE}")
        return False


    def _drop(self):
        rate = self.create_rate(10)
        clock = self.get_clock()

                
        # Move to drop pose
        ctrl = ArmControl()
        ctrl.header.stamp = clock.now().to_msg()
        ctrl.position = list(DEFAULT_DROP_POSE)
        ctrl.time = _to_uint16_list(DEFAULT_DROP_TIME_MS)
        self._control_pub.publish(ctrl)
        time_start = clock.now()
        while (clock.now() - time_start) < Duration(seconds=max(DEFAULT_LIFT_TIME_MS)*1.5 / 1000.0):
            rate.sleep()

        if np.allclose(DEFAULT_LIFT_POSE, self._current_position, atol=POSITION_TOLERANCE):
            return True
        self.get_logger().error("Failed to move to drop pose")
        self.get_logger().error(f"Current position: {self._current_position}")
        self.get_logger().error(f"Target position: {DEFAULT_LIFT_POSE}")


        drop_pos = DEFAULT_DROP_POSE.copy()
        drop_pos[0] = OPEN_GRIPPER_POSITION[0]
        drop_pos[1] = OPEN_GRIPPER_POSITION[1]
        # Close gripper
        ctrl = ArmControl()
        ctrl.header.stamp = clock.now().to_msg()
        ctrl.position = drop_pos
        ctrl.time = _to_uint16_list(OPEN_GRIPPER_TIME_MS * [.01, 1, 1, 1, 1, 1])
        self._control_pub.publish(ctrl)
        time_start = clock.now()
        while (clock.now() - time_start) < Duration(seconds=OPEN_GRIPPER_TIME_MS*1.5 / 1000.0):
            rate.sleep()

        if np.allclose(drop_pos, self._current_position, atol=POSITION_TOLERANCE):
            return True
        self.get_logger().error("Failed to open gripper")
        self.get_logger().error(f"Current position: {self._current_position}")
        self.get_logger().error(f"Target position: {drop_pos}")

        return False

    def _lift(self):
        rate = self.create_rate(10)
        clock = self.get_clock()

                
        # Move to pick pose
        ctrl = ArmControl()
        ctrl.header.stamp = clock.now().to_msg()
        ctrl.position = list(DEFAULT_LIFT_POSE)
        ctrl.time = _to_uint16_list(DEFAULT_LIFT_TIME_MS)
        self._control_pub.publish(ctrl)
        time_start = clock.now()
        while (clock.now() - time_start) < Duration(seconds=max(DEFAULT_LIFT_TIME_MS)*1.5 / 1000.0):
            rate.sleep()

        if np.allclose(DEFAULT_LIFT_POSE, self._current_position, atol=POSITION_TOLERANCE):
            return True
        self.get_logger().error("Failed to move to pick pose")
        self.get_logger().error(f"Current position: {self._current_position}")
        self.get_logger().error(f"Target position: {DEFAULT_LIFT_POSE}")
        return False
        
    def _pick_and_lift(self):
        if not self._pick():
            return False
        return self._lift()



def main(args=None):
    rclpy.init(args=args)
    node = ArmMoveActionServer()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
