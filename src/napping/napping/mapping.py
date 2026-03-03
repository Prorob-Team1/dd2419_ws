import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time

import rclpy.time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from nav_msgs.msg import OccupancyGrid
from napping.fov_tracking import FOVUpdater

from robp_interfaces.msg import (
    ObjectCandidateMsg,
    ObjectCandidateArrayMsg,
    ObjectDetectionMsg,
    ObjectDetectionArrayMsg,
)

import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import math
import uuid
import json


@dataclass
class Point2:
    x: float
    y: float


@dataclass
class Pose2D:
    x: float
    y: float
    angle: float


@dataclass
class ObjectSize:
    x: float
    y: float
    z: float


class ObjectClassification(Enum):
    CUBE_RED = "CUBE_R"
    CUBE_BLUE = "CUBE_B"
    CUBE_GREEN = "CUBE_G"
    CUBE_WOOD = "CUBE_W"
    CUBE_UNKNOWN = "CUBE_U"  # special case: given object without known color
    BOX = "BOX"


@dataclass
class ObjectCandidate:
    classification: ObjectClassification
    avg_pose: Pose2D
    log_prob: float
    count: int
    last_seen: Time
    id: str
    picked_up: bool


BOX_SIZE = ObjectSize(x=0.24, y=0.16, z=0.1)
CUBE_SIZE = ObjectSize(x=0.03, y=0.03, z=0.03)


class Mapper(Node):

    def __init__(self):
        super().__init__("mapper")
        self.get_logger().info("Mapper node started")

        self.occupancy_callback_group = MutuallyExclusiveCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.detection_sub = self.create_subscription(
            ObjectDetectionArrayMsg,
            "/object_detections",
            self.detection_callback,
            10,
        )
        self.pick_up_sub = self.create_subscription(
            ObjectCandidateArrayMsg,
            "/caught_cubes",
            self.picked_up_callback,
            10,
        )
        self.workspace_pub = self.create_publisher(
            Marker,
            "/geofence",
            10,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/occupancy_grid", 10)
        self.object_pub = self.create_publisher(
            ObjectCandidateArrayMsg, "/object_candidates", 10
        )
        self.object_pub_all = self.create_publisher(
            ObjectCandidateArrayMsg, "/object_candidates_raw", 10
        )
        self.object_marker_pub = self.create_publisher(
            Marker, "/object_candidate_markers", 10
        )

        # Periodic timer for simulating dynamic transform publishing
        # self.tf_timer = self.create_timer(0.05, self.publish_tf_map2odom)

        self.map_dir = Path("./maps")
        self.workspace_file = self.map_dir / "workspace_1.csv"
        self.map_file = self.map_dir / "map_1_1.csv"

        self.workspace: list[Point2] = []
        self.start_pose: Pose2D = Pose2D(0.0, 0.0, 0.0)
        self.given_boxes: list[Pose2D] = []
        self.given_objects: list[Pose2D] = []

        self.declare_parameter("resolution", 0.03)
        self.resolution: float = self.get_parameter("resolution").value  # type: ignore
        self.map_padding = [1, 1, 1.3, 1.2]
        self.occupancy_grid: OccupancyGrid

        self.object_candidates: list[ObjectCandidate] = []
        self.object_confidence_threshold = 0.8
        self.object_max_candidates = 100
        self.n_cubes = 3
        self.n_boxes = 2
        self.detection_mapper = DetectionMapper(self)
        self.fov_updater = FOVUpdater(self.tf_buffer, logger=self.get_logger())

        self.startup()

        # publishers
        self.occupancy_pub_timer = self.create_timer(
            0.5,
            self.publish_occupancy_map,
            callback_group=self.occupancy_callback_group,
        )
        self.fov_trace_timer = self.create_timer(
            0.2, self.apply_fov_trace, callback_group=self.occupancy_callback_group
        )
        self.marker_timer = self.create_timer(
            5.0, self.publish_workspace_perimeter_marker
        )
        self.object_timer = self.create_timer(0.1, self.publish_objects)
        self.candidate_display_timer = self.create_timer(
            0.1, self.display_object_candidates
        )
        # TODO: create service in the future instead of periodic export
        self.export_timer = self.create_timer(
            10.0, lambda: self.export_map(self.map_dir / "map_solution.json")
        )

    def startup(self):
        self.parse_workspace_file(self.workspace_file)
        self.parse_map_file(self.map_file)
        self.get_logger().info(
            f"Parsed workspace with {len(self.workspace)} points, {len(self.given_boxes)} boxes, {len(self.given_objects)} objects, start pose {self.start_pose}"
        )
        self.publish_workspace_perimeter_marker()
        self.publish_tf_map2odom()
        self.create_inital_object_candidates()
        self.occupancy_grid = self.create_occupancy_grid()

    def parse_map_file(self, file: Path):
        objects = []
        with open(file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Normalize fieldnames by stripping whitespace
            reader.fieldnames = [field.strip() for field in reader.fieldnames]  # type: ignore
            for row in reader:
                obj_type = row["Type"].strip()
                x = float(row["x"].strip()) / 100.0
                y = float(row["y"].strip()) / 100.0
                angle = float(row["angle"].strip())
                objects.append((obj_type, Pose2D(x, y, angle)))
                if obj_type == "S":
                    self.start_pose = Pose2D(x, y, angle)
                elif obj_type == "B":
                    self.given_boxes.append(Pose2D(x, y, angle))
                elif obj_type == "O":
                    self.given_objects.append(Pose2D(x, y, angle))
                else:
                    raise ValueError(f"Unknown object type '{obj_type}' in map file")
        return objects

    def parse_workspace_file(self, file: Path):
        polygon = []
        with open(file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                p = Point2(
                    x=float(row["x"].strip()) / 100.0, y=float(row["y"].strip()) / 100.0
                )
                polygon.append(p)
        self.workspace = polygon

    def publish_workspace_perimeter_marker(self):
        if len(self.workspace) < 3:
            raise ValueError("Workspace must have at least 3 points to form a polygon")

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "workspace"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02  # line width (meters)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Add polygon points
        points = [Point(x=p.x, y=p.y, z=0.0) for p in self.workspace]
        points.append(
            Point(x=self.workspace[0].x, y=self.workspace[0].y, z=0.0)
        )  # Close the polygon by adding the first point at the end
        marker.points = points
        self.workspace_pub.publish(marker)

    def publish_tf_map2odom(self):
        # TODO: will be dynamic with proper localization
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = self.start_pose.x
        t.transform.translation.y = self.start_pose.y
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(0, 0, self.start_pose.angle)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def point_in_polygon(self, x, y, polygon):
        """
        Check if point (x, y) is inside polygon using ray casting algorithm.
        """
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def create_occupancy_grid(self):
        """
        Create occupancy grid from workspace polygon and obstacles.
        """
        if not self.workspace:
            self.get_logger().error("No workspace polygon loaded!")
            raise ValueError("Workspace polygon is empty")

        # Find bounding box
        xs = [v.x for v in self.workspace]
        ys = [v.y for v in self.workspace]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Add padding
        min_x -= self.map_padding[0]
        min_y -= self.map_padding[1]
        max_x += self.map_padding[2]
        max_y += self.map_padding[3]

        # Calculate grid dimensions
        width = int((max_x - min_x) / self.resolution) + 1
        height = int((max_y - min_y) / self.resolution) + 1

        self.get_logger().info(f"Map size: {width}x{height} cells")
        self.get_logger().info(
            f"Map bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]"
        )

        # Create grid (0 = free, 100 = occupied, -1 = unknown)
        grid = np.full((height, width), 100, dtype=np.int8)

        # Vectorized point-in-polygon using ray casting
        cols = np.arange(width)
        rows = np.arange(height)
        col_grid, row_grid = np.meshgrid(cols, rows)
        x_coords = col_grid * self.resolution + min_x
        y_coords = row_grid * self.resolution + min_y

        inside = np.zeros((height, width), dtype=bool)
        polygon = self.workspace
        n = len(polygon)
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            cond1 = y_coords > min(p1y, p2y)
            cond2 = y_coords <= max(p1y, p2y)
            cond3 = x_coords <= max(p1x, p2x)
            if p1y != p2y:
                xinters = (y_coords - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            else:
                xinters = np.full_like(x_coords, p1x)
            cond4 = (p1x == p2x) | (x_coords <= xinters)
            mask = cond1 & cond2 & cond3 & cond4
            inside[mask] = ~inside[mask]
            p1x, p1y = p2x, p2y

        # Mark free cells inside workspace
        grid[inside] = -1  # unknown
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = min_x
        msg.info.origin.position.y = min_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # Flatten grid (row-major order)
        msg.data = grid.flatten().tolist()
        self.get_logger().info("Initial occupancy grid created")

        return msg

    def apply_fov_trace(self):
        # TODO: instead of passing occupancy_grid which always needs to be converted to numpy array,
        # we should pass the numpy array directly and only convert back to OccupancyGrid when publishing
        if self.occupancy_grid is not None:
            self.fov_updater.apply(self.occupancy_grid)

    def publish_occupancy_map(self):
        if self.occupancy_grid is not None:
            self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self.occupancy_grid)

    def create_inital_object_candidates(self):
        for box in self.given_boxes:
            self.object_candidates.append(
                ObjectCandidate(
                    classification=ObjectClassification.BOX,
                    avg_pose=Pose2D(box.x, box.y, box.angle),
                    log_prob=np.inf,
                    count=1,
                    last_seen=self.get_clock().now(),
                    id=str(uuid.uuid4()),
                    picked_up=False,
                )
            )
        for obj in self.given_objects:
            self.object_candidates.append(
                ObjectCandidate(
                    classification=ObjectClassification.CUBE_UNKNOWN,
                    avg_pose=Pose2D(obj.x, obj.y, obj.angle),
                    log_prob=np.inf,
                    count=1,
                    last_seen=self.get_clock().now(),
                    id=str(uuid.uuid4()),
                    picked_up=False,
                )
            )

    def publish_objects(self):
        arr_msg = ObjectCandidateArrayMsg()
        arr_msg.header.frame_id = "map"
        arr_msg.header.stamp = self.get_clock().now().to_msg()

        arr_msg_all = ObjectCandidateArrayMsg()
        arr_msg_all.header.frame_id = "map"
        arr_msg_all.header.stamp = self.get_clock().now().to_msg()

        for candidate in self.object_candidates:
            obj_msg = ObjectCandidateMsg()
            obj_msg.class_name = candidate.classification.value
            obj_msg.pose.position.x = candidate.avg_pose.x
            obj_msg.pose.position.y = candidate.avg_pose.y
            q = quaternion_from_euler(0, 0, candidate.avg_pose.angle)
            obj_msg.pose.orientation.x = q[0]
            obj_msg.pose.orientation.y = q[1]
            obj_msg.pose.orientation.z = q[2]
            obj_msg.pose.orientation.w = q[3]
            obj_msg.confidence = DetectionMapper.log_odds_to_probability(
                candidate.log_prob
            )
            obj_msg.picked_up = candidate.picked_up
            arr_msg_all.candidates.append(obj_msg)  # type: ignore
            if (
                DetectionMapper.log_odds_to_probability(candidate.log_prob)
                > self.object_confidence_threshold
            ):
                arr_msg.candidates.append(obj_msg)  # type: ignore
        self.object_pub.publish(arr_msg)
        self.object_pub_all.publish(arr_msg_all)

    def display_object_candidates(self):
        # use rviz markers
        for candidate in self.object_candidates:
            if (
                DetectionMapper.log_odds_to_probability(candidate.log_prob)
                > self.object_confidence_threshold
            ):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "object_candidates"
                marker.id = hash(candidate.id) & 0x7FFFFFFF
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = candidate.avg_pose.x
                marker.pose.position.y = candidate.avg_pose.y
                marker.pose.position.z = 0.01
                q = quaternion_from_euler(0, 0, candidate.avg_pose.angle)
                marker.pose.orientation.x = q[0]
                marker.pose.orientation.y = q[1]
                marker.pose.orientation.z = q[2]
                marker.pose.orientation.w = q[3]
                if candidate.classification == ObjectClassification.BOX:
                    marker.scale.x = BOX_SIZE.x
                    marker.scale.y = BOX_SIZE.y
                    marker.scale.z = BOX_SIZE.z
                    marker.color.r = 1.0
                    marker.color.g = 0.5
                    marker.color.b = 0.0
                    marker.color.a = 0.8
                else:
                    marker.scale.x = CUBE_SIZE.x
                    marker.scale.y = CUBE_SIZE.y
                    marker.scale.z = CUBE_SIZE.z
                    if candidate.classification == ObjectClassification.CUBE_RED:
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                    elif candidate.classification == ObjectClassification.CUBE_BLUE:
                        marker.color.r = 0.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0
                    elif candidate.classification == ObjectClassification.CUBE_GREEN:
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                    elif candidate.classification == ObjectClassification.CUBE_WOOD:
                        # yellow
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                    else:
                        # unknown magenta
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0

                    marker.color.a = 1.0

                self.object_marker_pub.publish(marker)

    def detection_callback(self, msg: ObjectDetectionArrayMsg):
        self.detection_mapper.process_object_detections(msg)

    def picked_up_callback(self, msg: ObjectCandidateArrayMsg):
        for candidate in self.object_candidates:
            for picked in msg.candidates:
                if candidate.id == picked.id:
                    candidate.picked_up = True
    def export_map(self, file: Path):
        cubes: list[ObjectCandidate] = []
        boxes: list[ObjectCandidate] = []
        for candidate in self.object_candidates:
            if candidate.classification == ObjectClassification.BOX:
                boxes.append(candidate)
            else:
                cubes.append(candidate)
        cubes.sort(key=lambda c: c.log_prob, reverse=True)
        boxes.sort(key=lambda c: c.log_prob, reverse=True)
        solution = {
            "cubes": [
                {
                    "class": c.classification.value,
                    "x": c.avg_pose.x,
                    "y": c.avg_pose.y,
                    "angle": c.avg_pose.angle,
                    "confidence": DetectionMapper.log_odds_to_probability(c.log_prob),
                }
                for c in cubes[: self.n_cubes]
            ],
            "boxes": [
                {
                    "x": b.avg_pose.x,
                    "y": b.avg_pose.y,
                    "angle": b.avg_pose.angle,
                    "confidence": DetectionMapper.log_odds_to_probability(b.log_prob),
                }
                for b in boxes[: self.n_boxes]
            ],
        }
        with open(file, "w") as f:
            json.dump(solution, f, indent=4)


class DetectionMapper:
    def __init__(self, node: Mapper, merge_threshold=0.1):
        self.node = node
        self.merge_threshold = merge_threshold
        self.log_prob_increase = 0.2
        self.log_prob_decrease = 0.1
        self.log_prob_init = self.probability_to_log_odds(0.5)

    @staticmethod
    def probability_to_log_odds(p):
        return math.log(p / (1 - p))

    @staticmethod
    def log_odds_to_probability(l):
        if l == np.inf:
            return 1.0
        if l == -np.inf:
            return 0.0
        return 1.0 / (1.0 + math.exp(-l))

    def process_object_detections(self, msg: ObjectDetectionArrayMsg):
        for detection in msg.detections:
            detection: ObjectDetectionMsg
            best_match = None
            min_dist = self.merge_threshold
            # check if close to any candidate aleady. If yes then update candidate
            for i, candidate in enumerate(self.node.object_candidates):
                same_class = detection.class_name == candidate.classification.value
                unknown_class = (
                    candidate.classification == ObjectClassification.CUBE_UNKNOWN
                )
                if not (same_class or unknown_class):
                    continue

                dist = np.linalg.norm(
                    [
                        detection.pose.position.x - candidate.avg_pose.x,
                        detection.pose.position.y - candidate.avg_pose.y,
                    ]
                )
                if dist < min_dist:
                    min_dist = dist
                    best_match = candidate
            _, _, yaw = euler_from_quaternion(
                [
                    detection.pose.orientation.x,
                    detection.pose.orientation.y,
                    detection.pose.orientation.z,
                    detection.pose.orientation.w,
                ]
            )

            if best_match is not None:
                best_match.count += 1
                if not best_match.log_prob == np.inf:  # no update for known objects
                    best_match.avg_pose.x += (
                        detection.pose.position.x - best_match.avg_pose.x
                    ) / best_match.count
                    best_match.avg_pose.y += (
                        detection.pose.position.y - best_match.avg_pose.y
                    ) / best_match.count

                    angle_diff = (yaw - best_match.avg_pose.angle + math.pi) % (
                        2 * math.pi
                    ) - math.pi
                    best_match.avg_pose.angle += angle_diff / best_match.count

                    # increase confidence (log odds) by a fixed amount (e.g. 0.5)
                    # TODO: scale with confidence of detection
                    best_match.log_prob += self.log_prob_increase
                best_match.last_seen = Time.from_msg(detection.header.stamp)
                # TODO: this is dependent on a single detection, should be changed in the future
                if (
                    best_match.classification.value
                    == ObjectClassification.CUBE_UNKNOWN.value
                ):
                    best_match.classification = ObjectClassification(
                        detection.class_name
                    )

            else:
                self.add_object_candidate(
                    ObjectCandidate(
                        classification=ObjectClassification(detection.class_name),
                        avg_pose=Pose2D(
                            detection.pose.position.x,
                            detection.pose.position.y,
                            yaw,
                        ),
                        log_prob=self.log_prob_init,
                        count=1,
                        last_seen=Time.from_msg(detection.header.stamp),
                        id=str(uuid.uuid4()),
                        picked_up=False,
                    )
                )

    def add_object_candidate(self, candidate: ObjectCandidate):
        # check if candidate is outside of the workspace, if yes then discard
        if not self.node.point_in_polygon(
            candidate.avg_pose.x,
            candidate.avg_pose.y,
            [(p.x, p.y) for p in self.node.workspace],
        ):
            self.node.get_logger().info(
                f"Discarding candidate {candidate.id} because it is outside of the workspace"
            )
            return
        if len(self.node.object_candidates) >= self.node.object_max_candidates:
            # remove lowest confidence candidate that is more than 10 seconds old
            now = self.node.get_clock().now()
            old_candidates = [
                c
                for c in self.node.object_candidates
                if (now - c.last_seen).nanoseconds > 10 * 1e9
            ]
            if old_candidates:
                worst_candidate = min(old_candidates, key=lambda c: c.log_prob)
                self.node.object_candidates.remove(worst_candidate)
            else:
                worst_candidate = min(
                    self.node.object_candidates, key=lambda c: c.log_prob
                )
                self.node.object_candidates.remove(worst_candidate)
            self.node.get_logger().warning(
                f"Had to remove candidate {worst_candidate.id} to make room for new candidate"
            )
        # add new candidate
        self.node.object_candidates.append(candidate)
        self.node.get_logger().info(
            f"Added new candidate with class {candidate.classification} at ({candidate.avg_pose.x:.2f}, {candidate.avg_pose.y:.2f}, {candidate.avg_pose.angle:.2f})"
        )
        assert len(self.node.object_candidates) <= self.node.object_max_candidates


def main():
    rclpy.init()
    node = Mapper()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
