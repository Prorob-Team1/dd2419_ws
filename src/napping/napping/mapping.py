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
from nav_msgs.msg import Path, OccupancyGrid

import numpy as np
import csv
import json
from pathlib import Path
from dataclasses import dataclass


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
        self.map_pub = self.create_publisher(OccupancyGrid, "/occupancy_grid", 10)

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
        self.occupancy_grid: OccupancyGrid

        # self.object_candidates: list[ObjectCandidate] = []

        self.startup()
        self.timer = self.create_timer(1.0, self.publish_map)
        self.marker_timer = self.create_timer(
            5.0, self.publish_workspace_perimeter_marker
        )

    def startup(self):
        self.parse_workspace_file(self.workspace_file)
        self.parse_map_file(self.map_file)
        self.get_logger().info(
            f"Parsed workspace with {len(self.workspace)} points, {len(self.given_boxes)} boxes, {len(self.given_objects)} objects, start pose {self.start_pose}"
        )
        self.publish_workspace_perimeter_marker()
        self.publish_tf_map2odom()
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
        padding = 0.5  # meters
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

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
        grid[inside] = 0
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

    def publish_map(self):
        if self.occupancy_grid is not None:
            self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self.occupancy_grid)


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
