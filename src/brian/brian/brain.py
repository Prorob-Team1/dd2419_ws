import py_trees

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
from rclpy.action.client import ActionClient
from rclpy.executors import MultiThreadedExecutor
from py_trees.composites import Sequence, Selector, Parallel
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from action_msgs.msg import GoalStatus

from robp_interfaces.action import DummyAction


class Brain(Node):

    def __init__(self):
        super().__init__("brain")
        self.tick_period = 0.1

        # Tree stuff

        # Conditions
        self.cube_found = False
        self.in_pickup_range = False
        self.cube_in_gripper = False

        self.in_dropoff_range = False
        #self.cube_released = False

        # Action clients

        self.explore_client = ActionClient(self, DummyAction, "dummy")

        self.nav_client = ActionClient(self, DummyAction, "dummy")

        self.arm_client = ActionClient(self, DummyAction, "dummy")

        self.select_client = ActionClient(self, DummyAction, "dummy")


        self.root = self.make_bt()
        self.create_timer(self.tick_period, self.root.tick_once)

    def make_bt(self):

        # Left subtree (catch cube)
        explore_fallback = Selector(
            name = "Explore Fallback",
            children = [
                CubeFoundCondition(self),
                ExploreB(self)
            ],
            memory = False
        )

        nav_to_cube_fallback = Selector(
            name = "Navigate to Cube Fallback",
            children = [
                InPickupRangeCondition(self),
                Nav2CubeB(self)
            ],
            memory = False
        )

        grab_cube_sequence = Sequence(
            name = "Grab Cube Sequence",
            children = [
                explore_fallback,
                nav_to_cube_fallback,
                GrabCubeB(self)
            ],
            memory = False
        )

        catch_fallback = Selector(
            name = "Catch Fallback",
            children = [
                CubePickedUpCondition(self),
                grab_cube_sequence
            ],
            memory = False
        )

        # Right subtree (release cube)
        nav_to_box_sequence = Sequence(
            name = "Nav to Box Sequence",
            children = [
                SelectBoxB(self),
                Nav2BoxB(self)
            ],
            memory = True # this could be a problem if we decide on a box the somehow becomes unreachable
        )

        box_in_drop_off_range_fallback = Selector(
            name = "Box in Range Fallback",
            children = [
                BoxInDropOffRangeCondition(self),
                nav_to_box_sequence
            ],
            memory = False
        )

        release_sequence = Sequence(
            name = "Release Cube Sequence",
            children  = [
                box_in_drop_off_range_fallback,
                ReleaseCubeB(self)
            ],
            memory = False
        )

        release_fallback = Selector(
            name = "Release Cube Fallback",
            children = [
                CubeReleasedCondition(self),
                release_sequence
            ],
            memory = False
        )



        # Connect both subtrees (catch and release)
        return Sequence(
            name="Root Sequence",
            children = [
                catch_fallback,
                release_fallback,
            ],
            memory = False
        )



class CubePickedUpCondition(Behaviour):

    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node

    def update(self):
        #self.node.get_logger().info("Checking if cube is picked up")
        if self.node.cube_in_gripper:
            return Status.SUCCESS
        else:
            return Status.FAILURE

class CubeFoundCondition(Behaviour):
    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node

    def update(self):
        #self.node.get_logger().info("Checking if cube has been found")
        if self.node.cube_found:
            return Status.SUCCESS
        else:
            return Status.FAILURE

class InPickupRangeCondition(Behaviour):

    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node
        
    def update(self):
        #self.node.get_logger().info("Checking if in pickup range")
        if self.node.in_pickup_range:
            return Status.SUCCESS
        else:
            return Status.FAILURE

class BoxInDropOffRangeCondition(Behaviour):

    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node

    def update(self):
        #self.node.get_logger().info("Checking if in dropoff range")
        if self.node.in_dropoff_range:
            return Status.SUCCESS
        else:
            return Status.FAILURE

class CubeReleasedCondition(Behaviour):

    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node

    def update(self):
        #self.node.get_logger().info("Checking if cube has been released")
        if not self.node.cube_in_gripper:
            return Status.SUCCESS
        else:
            return Status.FAILURE

class DummyB(Behaviour):

    def __init__(self, node: Brain, name, goal, action_client):
        super().__init__(name)
        self.node = node
        self.action_client = action_client
        self.goal = goal
        self.request_future = None
        self.goal_handle = None
        self.current_status = Status.RUNNING

    def update(self):
        #self.logger.info(f"{self.name}: Checking feedback")
        return self.current_status

    def terminate(self, new_status):
        if self.goal_handle is not None:
            self.logger.info(f"{self.name}: Interrupted, status: ")
            self.goal_handle.cancel_goal_async()


    def initialise(self):
        self.current_status = Status.RUNNING
        self.logger.info(f"{self.name}: Call some service")

        self.action_client.wait_for_server()
        goal = DummyAction.Goal(succeed=self.goal)
        self.request_future = self.action_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        self.request_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        pass
        #self.node.get_logger().info(f"{self.name}: Feedback: {feedback_msg.feedback.status}")

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.node.get_logger().info(f"{self.name}: Goal rejected")
            self.current_status = Status.FAILURE
            return
        self.node.get_logger().info(f"{self.name}: Goal accepted")
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.done_callback)
        
    def done_callback(self, future):
        try:
            response = future.result()
            result = response.result

            if response.status == GoalStatus.STATUS_SUCCEEDED:
                self.node.get_logger().info(f"{self.name}: Action goal succeeded! {result}")
                self.current_status = Status.SUCCESS
            else:
                self.node.get_logger().error(
                    f"{self.name}: Action goal failed with status: {response.status}"
                )
                self.current_status = Status.FAILURE
            
            self.goal_handle = None

        except Exception as e:
            self.node.get_logger().error(f"{self.name}: Action goal failed: {e}")
            self.current_status = Status.FAILURE

        self.update_postcondition()

    def update_postcondition(self):
        pass


class ExploreB(DummyB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, True, node.explore_client)
    
    def update_postcondition(self):
        if self.current_status == Status.SUCCESS:
            self.node.cube_found = True
        else:
            self.node.cube_found = False
        return 

class Nav2CubeB(DummyB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, True, node.nav_client)

    def update_postcondition(self):
        if self.current_status == Status.SUCCESS:
            self.node.in_pickup_range = True
        else:
            self.node.in_pickup_range = False
        return 

class Nav2BoxB(DummyB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, True, node.nav_client)

    def update_postcondition(self):
        if self.current_status == Status.SUCCESS:
            self.node.in_dropoff_range = True
        else:
            self.node.in_dropoff_range = False
        return 

class GrabCubeB(DummyB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, True, node.arm_client)

    def update_postcondition(self):
        if self.current_status == Status.SUCCESS:
            self.node.cube_in_gripper = True
        else:
            self.node.cube_in_gripper = False
        return 


class ReleaseCubeB(DummyB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, False, node.arm_client)

    def update_postcondition(self):
        if self.current_status == Status.SUCCESS:
            self.node.cube_in_gripper = False
            self.node.cube_found = False
            self.node.in_pickup_range = False
        else:
            self.node.cube_in_gripper = True
        return 

class SelectBoxB(DummyB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, True, node.select_client)


def main():
    rclpy.init()
    node = Brain()
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
