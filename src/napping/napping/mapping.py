import rclpy
import rclpy.duration
from rclpy.node import Node

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

import numpy as np
import csv
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Pose2D:
    x: float
    y: float
    angle: float


@dataclass
class ObjectSize:
    width: float
    length: float
    height: float


BOX_SIZE = ObjectSize(width=24, length=16, height=3)
CUBE_SIZE = ObjectSize(width=3, length=3, height=3)


class Mapper(Node):

    def __init__(self):
        super().__init__("mapper")
        self.get_logger().info("Mapper node started")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.workspace_pub = self.create_publisher(
            Marker,
            "/geofence",
            10,
        )
        # if you miss the initial publish, this will republish occasioanlly
        self.marker_timer = self.create_timer(
            5.0, self.publish_workspace_perimeter_marker
        )
        # Periodic timer for simulating dynamic transform publishing
        # self.tf_timer = self.create_timer(0.05, self.publish_tf_map2odom)

        self.map_dir = Path("./maps")
        self.workspace_file = self.map_dir / "workspace_1.csv"
        self.map_file = self.map_dir / "map_1_1.csv"

        self.workspace: list[Point] = []
        self.start_pose: Pose2D = Pose2D(0.0, 0.0, 0.0)
        self.given_boxes: list[Pose2D] = []
        self.given_objects: list[Pose2D] = []

        self.object_candidates: list[ObjectCandidate] = []

        # self.startup_timer = self.create_timer(0.0, self.startup)
        self.startup()

    def startup(self):
        self.parse_workspace_file(self.workspace_file)
        self.parse_map_file(self.map_file)
        self.get_logger().info(
            f"Parsed workspace with {len(self.workspace)} points, {len(self.given_boxes)} boxes, {len(self.given_objects)} objects, start pose {self.start_pose}"
        )
        self.publish_workspace_perimeter_marker()
        self.publish_tf_map2odom()

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
                p = Point()
                p.x = float(row["x"].strip()) / 100.0
                p.y = float(row["y"].strip()) / 100.0
                p.z = 0.0
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
        points = [p for p in self.workspace]
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


def main():
    rclpy.init()
    node = Mapper()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
