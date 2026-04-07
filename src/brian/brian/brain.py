#!/usr/bin/env python

import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
import rclpy.time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

from tf2_ros import TransformListener, Buffer
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from std_srvs.srv import Trigger
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseStamped, Point
from rclpy.action.client import ActionClient
from rclpy.executors import MultiThreadedExecutor

import py_trees
from py_trees.composites import Sequence, Selector
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from action_msgs.msg import GoalStatus
from std_msgs.msg import Bool

from robp_interfaces.action import DummyAction, Navigation
from robp_interfaces.msg import ObjectCandidateMsg,ObjectCandidateArrayMsg, DutyCycles

from napping.mapping import ObjectClassification

class ANSIEscClr():
    BOLD = "\x1b[1m"
    RESET = "\x1b[0m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    BLUE = "\x1b[94m"
    WOOD = "\x1b[33m"
    GRAY = "\x1b[90m"
    UNKNOWN = "\x1b[35m"


EXPLORE_GOAL = 0
CUBE_GOAL = 1
BOX_GOAL = 2

def format_goal_text(goal_type: int, target_cube: ObjectCandidateMsg):
    # Informative and fancy message string :)))))
    message = ""
    if goal_type == EXPLORE_GOAL:
        message = f"{ANSIEscClr.BOLD}EXPLORATION{ANSIEscClr.RESET}"
    elif goal_type == CUBE_GOAL:
        clr = ""
        if target_cube.class_name == ObjectClassification.CUBE_RED.value: clr = ANSIEscClr.RED
        elif target_cube.class_name == ObjectClassification.CUBE_GREEN.value: clr = ANSIEscClr.GREEN
        elif target_cube.class_name == ObjectClassification.CUBE_BLUE.value: clr = ANSIEscClr.BLUE
        elif target_cube.class_name == ObjectClassification.CUBE_WOOD.value: clr = ANSIEscClr.WOOD
        else: clr = ANSIEscClr.UNKNOWN
        message = f"{ANSIEscClr.BOLD}{clr}CUBE{ANSIEscClr.RESET}"
    elif goal_type == BOX_GOAL:
        message = f"{ANSIEscClr.BOLD}{ANSIEscClr.GRAY}BOX{ANSIEscClr.RESET}"
    return message    
    
class GoalProvider:
    """Provides exploration/object goals"""
    def __init__(self, logger, start_pose):
        self.logger = logger
        self.start_pose = start_pose
        self.target_cube = None

    def create_goal_marker(self, x: float, y: float, yaw: float, goal_type: int):
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        q = quaternion_from_euler(0.0, 0.0, yaw) if yaw != 0 else quaternion_from_euler(0.0, np.pi/2, 0.0) 
        goal_marker.pose.orientation.x = q[0]
        goal_marker.pose.orientation.y = q[1]
        goal_marker.pose.orientation.z = q[2]
        goal_marker.pose.orientation.w = q[3]
        goal_marker.ns = "goal"
        goal_marker.id = 0
        goal_marker.color.a = 1.0
        goal_marker.type = Marker.ARROW
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = x
        goal_marker.pose.position.y = y
        goal_marker.pose.position.z = 0.03
        goal_marker.scale.x = 0.2
        goal_marker.scale.y = 0.03
        goal_marker.scale.z = 0.03

        if goal_type == EXPLORE_GOAL:
            goal_marker.color.r = 0.0
            goal_marker.color.g = 1.0
            goal_marker.color.b = 0.0
            goal_marker.pose.position.z = 0.23
        elif goal_type == CUBE_GOAL:
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 0.0
        elif goal_type == BOX_GOAL:
            goal_marker.color.r = 0.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 1.0
        
        return goal_marker

    def get_goal(self, goal_type, robot_pose, map, valid_candidates, candidates):
        if goal_type == EXPLORE_GOAL: 
            self.logger.debug("Recieved exploration goal request")
            get_goal = lambda: self.get_explore_goal(robot_pose, map, candidates)
        elif goal_type == CUBE_GOAL:
            self.logger.debug("Recieved cube goal request")
            get_goal = lambda: self.get_object_goal(CUBE_GOAL, robot_pose, valid_candidates)
        elif goal_type == BOX_GOAL:
            self.logger.debug("Recieved box goal request")
            get_goal = lambda: self.get_object_goal(BOX_GOAL, robot_pose, valid_candidates)
        else:
            self.logger.warning("Recieved unknown goal request, returning an empty goal")
            return None
        
        if map == None:
            self.logger.warning("No map data available, returning an empty goal")
            return None
        
        try:
            (x,y,yaw) = get_goal()
            return x,y,yaw
        except TypeError as error:
            self.logger.warning(f"Oops, something went wrong: {error}")
        
        
        

    def find_frontiers(self, map_array):
        frontier_cells = []
        def is_frontier(i, j) -> bool:
            for k in range(i - 1, i + 2):
                for l in range(j - 1, j + 2):
                    if map_array[k][l] == -1:
                        return True
            return False
        h, w = map_array.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if map_array[i][j] == 0 and is_frontier(i, j):
                    frontier_cells.append((i, j))
        return frontier_cells

    def get_explore_goal(self, robot_pose, map_obj: OccupancyGrid, candidates):
        if map_obj is None or len(map_obj.data) < 1:
            self.logger.error("Recieved an empty map")
            return None
        map_arr = np.reshape(map_obj.data, (map_obj.info.height, map_obj.info.width))
        if robot_pose is None:
            return None

        r = np.random.rand()
        frontiers = np.array(self.find_frontiers(map_arr))
        if frontiers.size == 0:
            if len(candidates) < 1:
                self.logger.warning("No frontiers available, returning fallback goal (start position)")
                return self.start_pose 
            r = 1.0 # ensure we always go to a candidate

        # Go to a seen but uncertain object candidate 40% of the time if frontiers are available, otherwise every time
        if r > 0.6:
            if len(candidates) > 0:
                idx = np.random.choice(len(candidates))
                candidate = candidates[idx]
                x = candidate.pose.position.x
                y = candidate.pose.position.y
                yaw = euler_from_quaternion([
                    candidate.pose.orientation.x,
                    candidate.pose.orientation.y,
                    candidate.pose.orientation.z,
                    candidate.pose.orientation.w,
                ])[2]
                self.logger.debug(f"Sending goal at ({x=},{y=},{yaw=})")
                return x, y, 0.0 #yaw

        # Go to a new frontier
        robot_x, robot_y, robot_yaw = robot_pose
        grid_x = (robot_x - map_obj.info.origin.position.x) / map_obj.info.resolution
        grid_y = (robot_y - map_obj.info.origin.position.y) / map_obj.info.resolution
        grid_col = int(min(grid_x, map_obj.info.width - 1))
        grid_row = int(min(grid_y, map_obj.info.height - 1))
        
        rs = np.sum(([grid_row, grid_col] - frontiers) ** 2, axis=1)
        mask = (rs > 0.5 / map_obj.info.resolution)
        frontiers = frontiers[mask]
        if frontiers.size == 0:
            self.logger.warning("No distant frontiers available, returning fallback goal")
            return robot_x, robot_y, robot_yaw
        idx = np.random.choice(len(frontiers))
        row, col = frontiers[idx]
        x = col * map_obj.info.resolution + map_obj.info.origin.position.x
        y = row * map_obj.info.resolution + map_obj.info.origin.position.y
        yaw = np.atan2(y - robot_y, x - robot_x)
        self.logger.debug(f"Sending goal at ({x=},{y=},{yaw=})")
        return x, y, 0.0#yaw

    def get_object_goal(self, goal_type, robot_pose, valid_candidates):
        if robot_pose is None:
            return None
        valid_objects: list[ObjectCandidateMsg] = []
        for candidate in valid_candidates:
            if candidate.class_name == ObjectClassification.BOX.value and goal_type == BOX_GOAL:
                valid_objects.append(candidate)
            elif candidate.class_name != ObjectClassification.BOX.value and goal_type == CUBE_GOAL:
                valid_objects.append(candidate)
        closest_obj = None
        closest_pose = None
        closest_dist = np.inf
        for obj in valid_objects:
            q = [
                obj.pose.orientation.x,
                obj.pose.orientation.y,
                obj.pose.orientation.z,
                obj.pose.orientation.w,
            ]
            pose = [
                obj.pose.position.x,
                obj.pose.position.y,
                euler_from_quaternion(q)[2],
            ]
            new_dist = self.calc_dist(robot_pose, pose)
            if new_dist < closest_dist:
                if self.target_cube is not None:
                    if self.target_cube.id == obj.id:
                        continue
                closest_pose = pose
                closest_dist = new_dist
                closest_obj = obj
        if closest_obj is None:
            self.logger.warning("No valid object available, returning fallback goal")
            return robot_pose
        x, y, _ = closest_pose
        robot_x, robot_y, robot_yaw = robot_pose
        yaw = np.atan2(y - robot_y, x - robot_x)
        if goal_type == BOX_GOAL:
            q = [
                closest_obj.pose.orientation.x,
                closest_obj.pose.orientation.y,
                closest_obj.pose.orientation.z,
                closest_obj.pose.orientation.w
            ]
            yaw = euler_from_quaternion(q)[2] - np.pi/2 
        elif goal_type == EXPLORE_GOAL:
            yaw = 0

        self.logger.debug(f"Created object goal at (x={x:.2f},y={y:.2f},yaw={yaw:.2f})")
        if goal_type == CUBE_GOAL:
            self.target_cube = closest_obj
        return x, y, yaw

    def calc_dist(self, pose1: list[float], pose2: list[float]):
        x1, y1, yaw1 = pose1
        x2, y2, yaw2 = pose2
        return abs((x2 - x1) ** 2 + (y2 - y1) ** 2)

class Brain(Node):

    def __init__(self):
        super().__init__("brain")
        self.tick_period = 0.1

        # Use dummy behaviors for testing BT
        self.only_dummy_behaviors = False
        self.debugging = False

        # Mission info
        self.start_time = self.get_clock().now()
        self.mission_duration = Duration(seconds=300) # 5 minutes, then it's over :(

        # Conditions
        self.cube_found = False
        self.in_pickup_range = False
        self.cube_in_gripper = False
        self.in_dropoff_range = False

        self.has_backed_up = True

        # Map / object tracking
        self.map = None
        self.valid_candidates: list[ObjectCandidateMsg] = []
        self.potential_candidates: list[ObjectCandidateMsg] = []
        self.start_pose = None
        self.robot_stopped = True

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            "/occupancy_grid",
            self.map_callback,
            10
        )
        self.object_subscriber = self.create_subscription(
            ObjectCandidateArrayMsg,
            "/object_candidates",
            self.object_callback,
            10
        )

        self.candidate_sub = self.create_subscription(
            ObjectCandidateArrayMsg,
            "/object_candidates_raw",
            self.object_raw_callback,
            10
        )

        self.robot_in_motion_sub = self.create_subscription(
            DutyCycles,
            "/phidgets/motor/duty_cycles",
            self.robot_in_motion_callback,
            10
        )

        # Publishers
        self.goal_marker_publisher = self.create_publisher(Marker, "/nav_goal", 1)
        self.goal_obj_publisher = self.create_publisher(ObjectCandidateMsg, "/current_goal_obj", 10)
        self.caught_cubes_publisher = self.create_publisher(ObjectCandidateArrayMsg, "/caught_cubes", 10)
        detection_qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.detection_publisher = self.create_publisher(Bool, "/detection_on", detection_qos)
        self.detection_publisher.publish(Bool(data=True)) # turn on detection

        self.move_publisher = self.create_publisher(Point, "/move_dist", 10)

        # caught cubes
        self.caught_cubes: ObjectCandidateArrayMsg = ObjectCandidateArrayMsg()

        # TF for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Goal provider instance
        self.goal_provider = GoalProvider(logger=self.get_logger(), start_pose = self.start_pose)

        # Clients
        self.grabbing_client = self.create_client(Trigger, "/Start_Grasping")
        self.dropping_client = self.create_client(Trigger, "/Start_Dropping")

        self.dummy_arm_client = self.create_client(Trigger, "/dummy_service")

        # Action clients
        self.nav_client = ActionClient(self, Navigation, "plan_path")

        self.dummy_client = ActionClient(self, DummyAction, "dummy")


        self.root = self.make_bt()
        self.create_timer(self.tick_period, self.root.tick_once)

        self.get_logger().info(
            f"{ANSIEscClr.GREEN}{ANSIEscClr.BOLD}I AM ALIVE!{ANSIEscClr.RESET}"
        )

    def robot_in_motion_callback(self, msg: DutyCycles):
        if msg.duty_cycle_left == 0.0 and msg.duty_cycle_right == 0.0:
            self.robot_stopped = True
        else:
            self.robot_stopped = False

    def map_callback(self, msg: OccupancyGrid):
        self.map = msg
        if self.start_pose is None:
            self.start_pose = self.get_start_pose()

    def object_callback(self, msg: ObjectCandidateArrayMsg):
        # maintain list of valid candidates for use by goal_provider
        valid_candidates: list[ObjectCandidateMsg] = []
        self.cube_found = False
        for candidate in msg.candidates:
            candidate: ObjectCandidateMsg
            if candidate.picked_up == False or candidate.class_name == ObjectClassification.BOX.value:
                valid_candidates.append(candidate)
                if candidate.class_name != ObjectClassification.BOX.value:
                    self.cube_found = True
        self.valid_candidates = valid_candidates

    def object_raw_callback(self, msg: ObjectCandidateArrayMsg):
        # maintain list of uncertain candidates for use by goal_provider
        potential_candidates: list[ObjectCandidateMsg] = []
        for candidate in msg.candidates:
            candidate: ObjectCandidateMsg
            if not candidate.picked_up and candidate.confidence != 1.0:
                potential_candidates.append(candidate)
        self.potential_candidates = potential_candidates

    def update_caught_cubes(self):
        if self.goal_provider.target_cube is not None:
            self.caught_cubes.candidates.append(self.goal_provider.target_cube)
            self.caught_cubes.header.stamp = self.get_clock().now().to_msg()
            self.caught_cubes_publisher.publish(self.caught_cubes)

    def get_start_pose(self):
        if self.start_pose is not None:
            return self.start_pose
        if self.map is None:
            self.get_logger().error("Cannot compute pose without map timestamp")
            return None
        #stamp = self.map.header.stamp
        from_frame_rel = "odom"
        to_frame_rel = self.map.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=to_frame_rel,
                source_frame=from_frame_rel,
                time=Time(seconds=0), # latest TF available
                timeout=Duration(seconds=1)
            )
        except Exception as ex:
            self.get_logger().error(f"Couldn't find transform from {from_frame_rel} to {to_frame_rel}: {ex}")
            return None
        start_x = tf.transform.translation.x
        start_y = tf.transform.translation.y
        q = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
        start_yaw = euler_from_quaternion(q)[2]
        return (start_x, start_y, start_yaw)

    def get_robot_pose(self):
        if self.map is None:
            self.get_logger().error("Cannot compute pose without map timestamp")
            return None
        stamp = self.map.header.stamp
        from_frame_rel = "base_link"
        to_frame_rel = self.map.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=to_frame_rel,
                source_frame=from_frame_rel,
                time=Time(seconds=0),
                timeout=Duration(seconds=1)
            )
        except Exception as ex:
            self.get_logger().error(f"Couldn't find transform from {from_frame_rel} to {to_frame_rel}: {ex}")
            return None
        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y
        q = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
        robot_yaw = euler_from_quaternion(q)[2]
        return robot_x, robot_y, robot_yaw

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
        box_in_drop_off_range_fallback = Selector(
            name = "Box in Range Fallback",
            children = [
                BoxInDropOffRangeCondition(self),
                Nav2BoxB(self) # nav_to_box_sequence
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

        # backed up check
        backed_up_fallback = Selector(
            name="Backed Up Fallback",
            children = [
                BackedUpCondition(self),
                BackUpFromObjectB(self),
            ],
            memory = False
        )

        # Connect both subtrees (catch and release + backed up check)
        main_sequence = Sequence(
            name="Main Sequence",
            children = [
                backed_up_fallback,
                catch_fallback,
                release_fallback,
            ],
            memory = False
        )

        timer_sequence = Sequence(
            name="Timer Sequence",
            children = [
                TimeIsUpCondition(self),
                MakeResultsB(self),
            ],
            memory = False
        )

        # Main Fallback
        return Selector(
            name="Root Fallback",
            children = [
                timer_sequence,
                main_sequence,
            ],
            memory = False
        )

class BackedUpCondition(Behaviour):
    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node

    def update(self):
        if self.node.has_backed_up:
            return Status.SUCCESS
        else:
            if self.node.debugging:
                self.node.get_logger().info("fail back up")
            return Status.FAILURE

class CubePickedUpCondition(Behaviour):

    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node

    def update(self):
        #self.node.get_logger().info("Checking if cube is picked up")
        if self.node.cube_in_gripper:
            return Status.SUCCESS
        else:
            if self.node.debugging:
                self.node.get_logger().info("fail pick up")
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
            if self.node.debugging:
                self.node.get_logger().info("fail find cube")
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
            if self.node.debugging:
                self.node.get_logger().info("fail get close pick")
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
            if self.node.debugging:
                self.node.get_logger().info("fail get close drop")
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
            if self.node.debugging:
                self.node.get_logger().info("fail release")
            return Status.FAILURE
        
class TimeIsUpCondition(Behaviour):

    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node

    def update(self):
        mission_time = self.node.get_clock().now() - self.node.start_time
        if mission_time >= self.node.mission_duration:
            return Status.SUCCESS
        else:
            if self.node.debugging:
                self.node.get_logger().info("time ok")
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
            self.node.get_logger().debug(f"{self.name}: Interrupted, status: ")
            self.goal_handle.cancel_goal_async()


    def initialise(self):
        self.current_status = Status.RUNNING
        self.node.get_logger().debug(f"{self.name}: Call some service")

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
            self.node.get_logger().debug(f"{self.name}: Goal rejected")
            self.current_status = Status.FAILURE
            return
        self.node.get_logger().debug(f"{self.name}: Goal accepted")
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.done_callback)
        
    def done_callback(self, future):
        try:
            response = future.result()
            result = response.result

            if response.status == GoalStatus.STATUS_SUCCEEDED:
                self.node.get_logger().debug(f"{self.name}: Action goal succeeded! {result}")
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

class ArmB(Behaviour):
    def __init__(self, node: Brain, name, grabbing: bool):
        super().__init__(name)
        self.node = node
        self.current_status = Status.RUNNING
        self.grabbing = grabbing


    def terminate(self, new_status):
        self.node.get_logger().debug(f"Terminated arm action: {new_status}")
        self.node.detection_publisher.publish(Bool(data=True))
        pass # this could POTENTIALLY be a problem if we terminate in the middle of grasping

    def update(self):
        #self.logger.info(f"{self.name}: Checking feedback")
        return self.current_status
    
    def initialise(self):
        self.current_status = Status.RUNNING

        self.node.detection_publisher.publish(Bool(data=False))

        request = Trigger.Request()
        client = self.node.dropping_client
        if self.grabbing:
            client = self.node.grabbing_client
        if self.node.only_dummy_behaviors:
            client = self.node.dummy_arm_client
            
        client.wait_for_service(timeout_sec=1)
        if self.node.debugging:
            self.node.get_logger().info("Sent request to arm")
        future = client.call_async(request)
        future.add_done_callback(self.arm_callback)

    def arm_callback(self, future):
        response = future.result()
        if response is None:
            self.current_status = Status.FAILURE
            self.node.get_logger().warning("Never got a valid response from the arm.")
            return
        goal_msg = format_goal_text(CUBE_GOAL, self.node.goal_provider.target_cube)

        message = ""
        if response.success:
            self.current_status = Status.SUCCESS
            message = "Successfully dropped "
            if self.grabbing:
                message = "Successfully grabbed "
        else:
            self.current_status = Status.FAILURE
            message = "Failed to drop "
            if self.grabbing:
                message = "Failed to grab "
        message += goal_msg
        self.node.get_logger().info(message)
        self.update_postcondition()

    def update_postcondition(self):
        pass
        
class Nav2GoalB(Behaviour):

    def __init__(self, node: Brain, name, goal_type, done_status=Status.SUCCESS):
        super().__init__(name)
        self.node = node
        self.nav_goal_handle = None
        self.goal_type = goal_type
        self.current_status = Status.RUNNING
        self.done_status = done_status

    def update(self):
        #self.logger.info(f"{self.name}: Checking feedback")
        return self.current_status

    def terminate(self, new_status):
        if self.nav_goal_handle is not None:
            self.node.get_logger().debug(f"{self.name}: Interrupted, status: {new_status}")
            self.nav_goal_handle.cancel_goal_async()


    def initialise(self):
        
        self.current_status = Status.RUNNING
        if self.node.map is None:
            self.current_status = Status.FAILURE
            if self.node.debugging:
                self.node.get_logger().info(f"{self.name}: no map :(")
            return

        if self.node.debugging:
            self.node.get_logger().info(f"{self.name}: computing goal")
        result = self.node.goal_provider.get_goal(self.goal_type, self.node.get_robot_pose(), self.node.map, self.node.valid_candidates, self.node.potential_candidates)
        if result is None:
            self.node.get_logger().error(f"{self.name}: goal provider failed.")
            self.current_status = Status.FAILURE
            return
        x, y, yaw = result
        # publish marker for debugging
        marker = self.node.goal_provider.create_goal_marker(x, y, yaw, self.goal_type)
        marker.header.stamp = self.node.get_clock().now().to_msg()
        self.node.goal_marker_publisher.publish(marker)

        # Informative and fancy message :)))))
        goal_str = format_goal_text(self.goal_type, self.node.goal_provider.target_cube)
        self.node.get_logger().info(
            f"Sent {goal_str} goal at {ANSIEscClr.BOLD}({x=:.2f}, {y=:.2f}){ANSIEscClr.RESET}"
        )
        goal = Navigation.Goal()
        goal.goal.pose.position.x = x
        goal.goal.pose.position.y = y
        q = quaternion_from_euler(0.0,0.0,yaw)
        
        best_candidate = None
        if self.goal_type == CUBE_GOAL and self.node.goal_provider.target_cube is not None:
            goal.goal_label = self.node.goal_provider.target_cube.class_name
            best_candidate = self.node.goal_provider.target_cube
        elif self.goal_type == BOX_GOAL:
            min_dist = np.inf
            for candidate in self.node.valid_candidates:
                if candidate.class_name != ObjectClassification.BOX.value:
                    continue
                x_c = candidate.pose.position.x
                y_c = candidate.pose.position.y
                dist = (x-x_c)**2 + (y-y_c)**2 
                if dist < min_dist:
                    min_dist = dist
                    goal.goal_label = candidate.class_name
                    best_candidate = candidate
        if best_candidate is not None:
            self.node.goal_obj_publisher.publish(best_candidate)
        else:
            self.node.goal_obj_publisher.publish(ObjectCandidateMsg(id=""))
        
        time.sleep(1) # let the map inflator update before sending goal
        goal.goal.pose.orientation.z = q[2]
        goal.goal.pose.orientation.w = q[3]
        goal.goal.header.frame_id = "map"
        goal.goal.header.stamp = self.node.get_clock().now().to_msg()
        #if self.goal_type == BOX_GOAL:
        #    goal.goal.goal_label = "BOX"
        #elif self.goal_type == CUBE_GOAL:
        #    goal.goal.goal_label = "CUBE"
        if self.node.only_dummy_behaviors:
            self.node.dummy_client.wait_for_server()
            nav_request_future = self.node.dummy_client.send_goal_async(
                DummyAction.Goal(succeed=True), feedback_callback=self.nav_feedback_callback
            )
        else:
            self.node.nav_client.wait_for_server()
            nav_request_future = self.node.nav_client.send_goal_async(
                goal, feedback_callback=self.nav_feedback_callback
            )
        nav_request_future.add_done_callback(self.nav_goal_response_callback)


    def nav_feedback_callback(self, feedback_msg):
        pass
        #self.node.get_logger().info(f"{self.name}: Feedback: {feedback_msg.feedback.status}")

    def nav_goal_response_callback(self, future):
        self.nav_goal_handle = future.result()
        if not self.nav_goal_handle.accepted:
            self.node.get_logger().warning(f"Navigation goal rejected")
            self.current_status = Status.FAILURE
            return
        self.node.get_logger().debug(f"{self.name}: Navigation goal accepted")
        result_future = self.nav_goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_done_callback)

        goal_str = format_goal_text(self.goal_type, self.node.goal_provider.target_cube)
        message = f"--> Navigating to {goal_str}"
        if self.goal_type == EXPLORE_GOAL:
            message += "-goal"
        self.node.get_logger().info(message)
        
    def nav_done_callback(self, future):
        try:
            response = future.result()
            result = response.result

            if response.status == GoalStatus.STATUS_SUCCEEDED:
                self.node.get_logger().info(f"--> DONE!")
                self.current_status = self.done_status
            else:
                self.node.get_logger().error(
                    f"{self.name}: Navigation failed with status: {response.status}"
                )
                self.current_status = Status.FAILURE
            
            self.nav_goal_handle = None

        except Exception as e:
            self.node.get_logger().error(f"{self.name}: Navigation failed: {e}")
            self.current_status = Status.FAILURE
        
        self.update_postcondition()

    def update_postcondition(self):
        pass

class ExploreB(Nav2GoalB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, EXPLORE_GOAL, Status.FAILURE)

class Nav2CubeB(Nav2GoalB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, CUBE_GOAL)

    def update_postcondition(self):
        self.node.in_dropoff_range = False
        if self.current_status == Status.SUCCESS:
            self.node.in_pickup_range = True
        else:
            self.node.in_pickup_range = False

class Nav2BoxB(Nav2GoalB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, BOX_GOAL)

    def update_postcondition(self):
        self.node.in_pickup_range = False
        if self.current_status == Status.SUCCESS:
            self.node.in_dropoff_range = True
        else:
            self.node.in_dropoff_range = False

class GrabCubeB(ArmB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, True)

    def update_postcondition(self):
        if self.current_status == Status.SUCCESS:
            self.node.update_caught_cubes()
            self.node.cube_in_gripper = True
        else:
            self.node.cube_in_gripper = False
        self.node.has_backed_up = False 
        self.node.detection_publisher.publish(Bool(data=True))

class ReleaseCubeB(ArmB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, False)

    def update_postcondition(self):
        if self.current_status == Status.SUCCESS:
            self.node.cube_in_gripper = False
        else:
            self.node.cube_in_gripper = True
        self.node.has_backed_up = False
        self.node.detection_publisher.publish(Bool(data=True))
        
class MoveRobotB(Behaviour):
    def __init__(self, node: Brain, name, x_distance: float, y_distance: float = 0.0):
        super().__init__(name)
        self.node = node
        self.current_status = Status.RUNNING
        self.x_distance = x_distance
        self.y_distance = y_distance

        self.init_time = 0


    def terminate(self, new_status):
        self.node.get_logger().debug(f"Terminated arm action: {new_status}")
        pass

    def update(self):
        #self.logger.info(f"{self.name}: Checking feedback")
        time_diff = self.node.get_clock().now().to_msg().sec - self.init_time
        if time_diff > 1 and self.node.robot_stopped:
            # We wait until it's been at least 1 second before checking if backup is complete
            self.current_status = Status.SUCCESS
            self.update_postcondition()
        return self.current_status
    
    def initialise(self):
        self.current_status = Status.RUNNING
        self.init_time = self.node.get_clock().now().to_msg().sec
        msg = Point()
        msg.x = self.x_distance
        msg.y = self.y_distance
        self.log_action()
        self.node.move_publisher.publish(msg)

    def log_action(self):
        pass

    def update_postcondition(self):
        pass
    
class BackUpFromObjectB(MoveRobotB):
    def __init__(self, node: Brain):
        super().__init__(node, __class__.__name__, x_distance=-0.5, y_distance=0.0)
    def log_action(self):
        self.node.get_logger().info("--> Backing up...")

    def update_postcondition(self):
        if self.current_status == Status.SUCCESS:
            self.node.get_logger().info("--> DONE!")
            self.node.has_backed_up = True
            self.node.in_dropoff_range = False
            self.node.in_pickup_range = False
        else:
            self.node.has_backed_up = False
        
class MakeResultsB(Behaviour):
    def __init__(self, node: Brain):
        super().__init__(__class__.__name__)
        self.node = node
        self.current_status = Status.RUNNING

    def update(self):
        return self.current_status

    def terminate(self, new_status):
        pass


    def initialise(self):
        self.node.get_logger().info("TIME'S UP, MAGGOT")
        pass

def main():
    rclpy.init()
    node = Brain()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.get_logger().info(f"{ANSIEscClr.RED}{ANSIEscClr.BOLD}I AM DEAD!{ANSIEscClr.RESET}")
    executor.shutdown(timeout_sec=1)
    rclpy.shutdown()


if __name__ == "__main__":
    main()