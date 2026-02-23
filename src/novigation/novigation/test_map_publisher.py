import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import csv
from pathlib import Path


class TestMapPublisher(Node):
    """
    Publishes a static occupancy grid map based on workspace polygon and obstacles.
    Used for testing path planning.
    """

    def __init__(self):
        super().__init__('test_map_publisher')

        # Parameters
        self.declare_parameter('workspace_file', '/home/robot/dd2419_ws/src/novigation/novigation/workspace_1.csv')
        self.declare_parameter('map_file', '/home/robot/dd2419_ws/src/novigation/novigation/map_1_1.csv')
        self.declare_parameter('csv_scale', 0.01)  # CSV units to meters (cm -> m)
        self.declare_parameter('resolution', 0.03)  # Grid cell size in meters
        self.declare_parameter('obstacle_radius', 3)  # Grid cells around obstacles

        workspace_file = self.get_parameter('workspace_file').value
        map_file = self.get_parameter('map_file').value
        self.csv_scale = self.get_parameter('csv_scale').value
        self.resolution = self.get_parameter('resolution').value
        self.obstacle_radius = self.get_parameter('obstacle_radius').value

        # Load data
        self.workspace_polygon = self.load_workspace(workspace_file)
        self.obstacles = self.load_obstacles(map_file)

        # Create occupancy grid
        self.occupancy_grid = self.create_occupancy_grid()

        # Publisher
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # Publish map at 1Hz
        self.timer = self.create_timer(1.0, self.publish_map)

        self.get_logger().info('Test Map Publisher initialized')
        self.get_logger().info(f'Workspace vertices: {len(self.workspace_polygon)}')
        self.get_logger().info(f'Obstacles: {len(self.obstacles)}')

    def load_workspace(self, filepath):
        """Load workspace polygon from CSV."""
        vertices = []
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    x = float(row['x'].strip()) * self.csv_scale
                    y = float(row['y'].strip()) * self.csv_scale
                    vertices.append((x, y))
            self.get_logger().info(f'Loaded {len(vertices)} workspace vertices')
        except Exception as e:
            self.get_logger().error(f'Failed to load workspace: {e}')
        return vertices

    def load_obstacles(self, filepath):
        """Load obstacles from map CSV."""
        obstacles = []
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    obj_type = row['Type'].strip()
                    x = float(row['x'].strip())
                    y = float(row['y'].strip())

                    # Only add obstacles (type 'O') and boxes (type 'B')
                    if obj_type in ['O', 'B']:
                        obstacles.append((x * self.csv_scale, y * self.csv_scale))
                        self.get_logger().info(f'Obstacle at ({x * self.csv_scale}, {y * self.csv_scale})')

            self.get_logger().info(f'Loaded {len(obstacles)} obstacles')
        except Exception as e:
            self.get_logger().error(f'Failed to load obstacles: {e}')
        return obstacles

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
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def create_occupancy_grid(self):
        """
        Create occupancy grid from workspace polygon and obstacles.
        """
        if not self.workspace_polygon:
            self.get_logger().error('No workspace polygon loaded!')
            return None

        # Find bounding box
        xs = [v[0] for v in self.workspace_polygon]
        ys = [v[1] for v in self.workspace_polygon]

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

        self.get_logger().info(f'Map size: {width}x{height} cells')
        self.get_logger().info(f'Map bounds: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]')

        # Create grid (0 = free, 100 = occupied, -1 = unknown)
        grid = np.full((height, width), 100, dtype=np.int8)

        # Vectorized point-in-polygon using ray casting
        cols = np.arange(width)
        rows = np.arange(height)
        col_grid, row_grid = np.meshgrid(cols, rows)
        x_coords = col_grid * self.resolution + min_x
        y_coords = row_grid * self.resolution + min_y

        inside = np.zeros((height, width), dtype=bool)
        polygon = self.workspace_polygon
        n = len(polygon)
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
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

        # Add obstacles
        for obs_x, obs_y in self.obstacles:
            # Convert to grid coordinates
            col = int((obs_x - min_x) / self.resolution)
            row = int((obs_y - min_y) / self.resolution)

            # Mark obstacle region (circular)
            for dr in range(-self.obstacle_radius, self.obstacle_radius + 1):
                for dc in range(-self.obstacle_radius, self.obstacle_radius + 1):
                    if dr*dr + dc*dc <= self.obstacle_radius*self.obstacle_radius:
                        r = row + dr
                        c = col + dc
                        if 0 <= r < height and 0 <= c < width:
                            grid[r, c] = 100

        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = min_x
        msg.info.origin.position.y = min_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # Flatten grid (row-major order)
        msg.data = grid.flatten().tolist()

        return msg

    def publish_map(self):
        """Publish the map."""
        if self.occupancy_grid is not None:
            self.occupancy_grid.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self.occupancy_grid)


def main(args=None):
    rclpy.init(args=args)
    node = TestMapPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
