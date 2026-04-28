import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from robp_interfaces.msg import ObjectCandidateArrayMsg, ObjectCandidateMsg

import numpy as np
import scipy.ndimage as ndimage
from math import sqrt


BOX_HALF_X = 0.15   
BOX_HALF_Y = 0.10   


class MapInflator(Node):

    def __init__(self):
        super().__init__('map_inflator')

        self.inflation_radius_m = 0.12         
        self.pseudo_hard_radius_m = 0.20       
        self.cost_inflation_radius_m = 0.40    
        self.box_cost_radius_m = 1.0           

        self.base_grid = None
        self.candidates = []
        self.goal_candidate = None

        self.create_subscription(
            OccupancyGrid, '/occupancy_grid', self.grid_callback, 10
        )
        self.create_subscription(
            ObjectCandidateArrayMsg, '/object_candidates', self.candidates_callback, 10
        )
        self.create_subscription(
            ObjectCandidateMsg, '/current_goal_obj', self.goal_candidate_callback, 10
        )

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.create_timer(0.2, self._rebuild_and_publish)  

        self.get_logger().info(
            f'Map inflator initialized. Hard={self.inflation_radius_m}m, Plateau={self.pseudo_hard_radius_m}m'
        )

    def grid_callback(self, msg: OccupancyGrid):
        self.base_grid = msg

    def candidates_callback(self, msg: ObjectCandidateArrayMsg):
        self.candidates = list(msg.candidates)

    def goal_candidate_callback(self, msg: ObjectCandidateMsg):
        self.goal_candidate = msg

    def _yaw_from_quat(self, q):
        return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _paint_box_corridor_cost(self, grid, candidate, resolution, origin_x, origin_y, width, height, inflated_mask):
        cx = candidate.pose.position.x
        cy = candidate.pose.position.y
        yaw = self._yaw_from_quat(candidate.pose.orientation)

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        radius_cells = int(self.box_cost_radius_m / resolution)
        col_center = int((cx - origin_x) / resolution)
        row_center = int((cy - origin_y) / resolution)

        r_lo = max(0, row_center - radius_cells)
        r_hi = min(height - 1, row_center + radius_cells)
        c_lo = max(0, col_center - radius_cells)
        c_hi = min(width - 1, col_center + radius_cells)

        cols = np.arange(c_lo, c_hi + 1)
        rows = np.arange(r_lo, r_hi + 1)
        cc, rr = np.meshgrid(cols, rows)

        dx = cc * resolution + origin_x - cx
        dy = rr * resolution + origin_y - cy

        local_x = dx * cos_y + dy * sin_y
        local_y = -dx * sin_y + dy * cos_y

        dist = np.hypot(dx, dy)
        max_dist = radius_cells * resolution

        region = grid[r_lo:r_hi + 1, c_lo:c_hi + 1]
        region_inflated = inflated_mask[r_lo:r_hi + 1, c_lo:c_hi + 1]
        free = (~region_inflated) & (dist <= max_dist) & (dist > 1e-6)

        corridor_half_length = BOX_HALF_X/2
        fade_width = 0.2

        abs_local_x = np.abs(local_x)

        in_corridor = abs_local_x < corridor_half_length
        in_fade = (abs_local_x >= corridor_half_length) & (abs_local_x < corridor_half_length + fade_width)
        in_block = abs_local_x >= corridor_half_length + fade_width

        fade_cost = (80 * (abs_local_x - corridor_half_length) / fade_width).astype(np.int8)

        block_mask = free & in_block & (region < 80)
        region[block_mask] = 80

        fade_mask = free & in_fade & (fade_cost > region)
        region[fade_mask] = fade_cost[fade_mask]

        clear_mask = free & in_corridor & (region > 0) & (region < 100)
        region[clear_mask] = 0

    def _rebuild_and_publish(self):
        if self.base_grid is None:
            return
        t0 = self.get_clock().now()

        info = self.base_grid.info
        resolution = info.resolution
        width = info.width
        height = info.height
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        grid = np.array(self.base_grid.data, dtype=np.int8).reshape((height, width))

        box_candidates = []

        for candidate in self.candidates:
            if self.goal_candidate is not None:
                if candidate.id == self.goal_candidate.id:
                    if candidate.class_name == 'BOX':
                        box_candidates.append(candidate)
                    continue
            
            if candidate.picked_up:
                continue

            cx = candidate.pose.position.x
            cy = candidate.pose.position.y

            if candidate.class_name == 'BOX':
                col_min = max(0, int((cx - BOX_HALF_X - origin_x) / resolution))
                col_max = min(width - 1, int((cx + BOX_HALF_X - origin_x) / resolution))
                row_min = max(0, int((cy - BOX_HALF_Y - origin_y) / resolution))
                row_max = min(height - 1, int((cy + BOX_HALF_Y - origin_y) / resolution))
                grid[row_min:row_max + 1, col_min:col_max + 1] = 100
                
            else:
                col = int((cx - origin_x) / resolution)
                row = int((cy - origin_y) / resolution)
                if 0 <= row < height and 0 <= col < width:
                    grid[row, col] = 100

        base_empty_space = (grid != 100)

        dist_pixels = ndimage.distance_transform_edt(base_empty_space)
        dist_m = dist_pixels * resolution

        hard_mask = base_empty_space & (dist_m <= self.inflation_radius_m)
        grid[hard_mask] = 100

        plateau_mask = base_empty_space & (dist_m > self.inflation_radius_m) & (dist_m <= self.pseudo_hard_radius_m)
        safe_plateau_mask = plateau_mask & (grid < 99)
        grid[safe_plateau_mask] = 99

        decay_mask = base_empty_space & (dist_m > self.pseudo_hard_radius_m) & (dist_m <= self.cost_inflation_radius_m)
        
        decay_dist = dist_m[decay_mask] - self.pseudo_hard_radius_m
        decay_max = self.cost_inflation_radius_m - self.pseudo_hard_radius_m

        if decay_max > 0:
            fraction = decay_dist / decay_max
            raw_costs = 99.0 * ((1.0 - fraction) ** 2.0)
            
            new_costs = np.clip(raw_costs, 1, 99).astype(np.int8)
            existing_costs = grid[decay_mask]
            
            grid[decay_mask] = np.maximum(existing_costs, new_costs)

        inflated_mask = (grid == 100)

        for candidate in box_candidates:
            self._paint_box_corridor_cost(
                grid, candidate, resolution, origin_x, origin_y, width, height, inflated_mask
            )

        out = OccupancyGrid()
        out.header.frame_id = self.base_grid.header.frame_id
        out.header.stamp = self.get_clock().now().to_msg()
        out.info = info
        out.data = grid.flatten().tolist()
        self.map_pub.publish(out)
        
        dt_ms = (self.get_clock().now() - t0).nanoseconds / 1e6
        self.get_logger().info(f'Map inflated and published in {dt_ms:.1f} ms', throttle_duration_sec=5.0)


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