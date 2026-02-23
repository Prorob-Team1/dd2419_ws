import math
import numpy as np
import rclpy.duration
import rclpy.time
from nav_msgs.msg import OccupancyGrid
from tf2_ros.buffer import Buffer
from tf_transformations import euler_from_quaternion


class CameraTrapezoid:
    def __init__(self, opening_angle: float, r_min: float, r_max: float):
        self.alpha = opening_angle / 2.0
        self.r_min = r_min
        self.r_max = r_max
        self.tan_alpha = math.tan(self.alpha)

    def vertices_robot(self) -> np.ndarray:
        w_min = self.r_min * self.tan_alpha
        w_max = self.r_max * self.tan_alpha
        return np.array(
            [
                [self.r_min, -w_min],
                [self.r_min, w_min],
                [self.r_max, w_max],
                [self.r_max, -w_max],
            ]
        )


class FOVUpdater:
    """Applies camera FOV clearing to an OccupancyGrid in place.

    Designed to be owned by a ROS node. Pass the node's tf_buffer and
    logger on construction, then call apply() periodically.
    """

    def __init__(
        self,
        tf_buffer: Buffer,
        logger=None,
        # TODO: These might cahgne dynamically based on occlusion
        opening_angle: float = math.radians(80),
        r_min: float = 0.2,
        r_max: float = 1.5,
    ):
        self.tf_buffer = tf_buffer
        self.logger = logger
        self.trapezoid = CameraTrapezoid(opening_angle, r_min, r_max)

    def apply(self, occupancy_grid: OccupancyGrid) -> bool:
        """Clear cells inside the camera FOV trapezoid. Modifies grid in place.

        Returns True on success, False if the TF lookup failed.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as e:
            if self.logger:
                self.logger.warn(
                    f"FOV TF lookup failed: {e}", throttle_duration_sec=5.0
                )
            return False

        # Extract robot pose in map frame
        cam_x = t.transform.translation.x
        cam_y = t.transform.translation.y
        cam_theta = euler_from_quaternion(
            [
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ]
        )[2]

        c, s = math.cos(cam_theta), math.sin(cam_theta)
        R = np.array([[c, -s], [s, c]])

        # Trapezoid bounding box in map frame
        verts_map = self.trapezoid.vertices_robot() @ R.T + np.array([cam_x, cam_y])
        min_x, min_y = verts_map.min(axis=0)
        max_x, max_y = verts_map.max(axis=0)

        # Grid parameters
        res = occupancy_grid.info.resolution
        origin_x = occupancy_grid.info.origin.position.x
        origin_y = occupancy_grid.info.origin.position.y
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height

        gx0 = max(0, int((min_x - origin_x) / res))
        gy0 = max(0, int((min_y - origin_y) / res))
        gx1 = min(width - 1, int((max_x - origin_x) / res))
        gy1 = min(height - 1, int((max_y - origin_y) / res))

        if gx0 > gx1 or gy0 > gy1:
            return True  # trapezoid outside map — not an error

        # Build offset vectors over the bounding box (broadcast instead of meshgrid)
        xs = origin_x + (np.arange(gx0, gx1 + 1) + 0.5) * res
        ys = origin_y + (np.arange(gy0, gy1 + 1) + 0.5) * res

        dx = xs[None, :] - cam_x  # shape (1, W)
        dy = ys[:, None] - cam_y  # shape (H, 1)

        x_c = c * dx + s * dy
        y_c = -s * dx + c * dy
        mask = (
            (x_c >= self.trapezoid.r_min)
            & (x_c <= self.trapezoid.r_max)
            & (np.abs(y_c) <= x_c * self.trapezoid.tan_alpha)
        )

        # Zero out unknown cells (keep walls at 100, keep already-free at 0)
        grid = np.array(occupancy_grid.data).reshape((height, width))
        subgrid = grid[gy0 : gy1 + 1, gx0 : gx1 + 1]
        subgrid[mask & (subgrid != 100)] = 0
        occupancy_grid.data = grid.flatten().tolist()

        return True
