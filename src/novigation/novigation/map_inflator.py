import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from robp_interfaces.msg import ObjectCandidateArrayMsg

import numpy as np



BOX_HALF_X = 0.12   
BOX_HALF_Y = 0.08   

class MapInflator(Node):

    def __init__(self):
        super().__init__('map_inflator')

      
        self.inflation_radius_m = 0.175

        self.base_grid = None    
        self.candidates = []     

        self.create_subscription(
            OccupancyGrid, '/occupancy_grid', self.grid_callback, 10
        )
        self.create_subscription(
            ObjectCandidateArrayMsg, '/object_candidates', self.candidates_callback, 10
        )

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.get_logger().info(
            f'Map inflator initialized, inflation_radius={self.inflation_radius_m:.3f}m'
        )

    def grid_callback(self, msg: OccupancyGrid):
        self.base_grid = msg
        self._rebuild_and_publish()

    def candidates_callback(self, msg: ObjectCandidateArrayMsg):
        self.candidates = list(msg.candidates)
        self._rebuild_and_publish()

    def _rebuild_and_publish(self):
        if self.base_grid is None:
            return

        info = self.base_grid.info
        resolution = info.resolution
        width = info.width
        height = info.height
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        grid = np.array(self.base_grid.data, dtype=np.int8).reshape((height, width))

        
        for candidate in self.candidates:
            
            cx = candidate.pose.position.x
            
            cy = candidate.pose.position.y

            if candidate.class_name == 'BOX':
                col_min = max(0, int((cx - BOX_HALF_X - origin_x) / resolution))
                col_max = min(width - 1, int((cx + BOX_HALF_X - origin_x) / resolution))
                
                row_min = max(0, int((cy - BOX_HALF_Y - origin_y) / resolution))
                row_max = min(height - 1, int((cy + BOX_HALF_Y - origin_y) / resolution))
                
                grid[row_min:row_max + 1, col_min:col_max + 1] = 100
            else:
                # cubes
                col = int((cx - origin_x) / resolution)
                
                row = int((cy - origin_y) / resolution)
                if 0 <= row < height and 0 <= col < width:
                    grid[row, col] = 100

        # Inflate all occupied cells by robot radius
        radius_cells = max(1, int(self.inflation_radius_m / resolution))
        occupied = grid == 100
        inflated = occupied.copy()
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                shifted = np.roll(occupied, (dr, dc), axis=(0, 1))
                # Clear wrapped edges so objects dont inflate across the opposite side of the map
                if dr > 0:
                    shifted[:dr, :] = False
                elif dr < 0:
                    shifted[dr:, :] = False
                if dc > 0:
                    shifted[:, :dc] = False
                elif dc < 0:
                    shifted[:, dc:] = False
                inflated |= shifted
        grid[inflated] = 100

        
        out = OccupancyGrid()
        out.header.frame_id = self.base_grid.header.frame_id
        out.header.stamp = self.get_clock().now().to_msg()
        out.info = info
        out.data = grid.flatten().tolist()
        self.map_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MapInflator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
