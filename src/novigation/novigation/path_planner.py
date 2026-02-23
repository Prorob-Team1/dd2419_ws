import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros import TransformListener, Buffer

from robp_interfaces.action import Navigation
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped as PathPose

from math import sqrt
import heapq


class PathPlannerNode(Node):
  

    def __init__(self):
        super().__init__('path_planner')

        
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('planning_timeout', 5.0)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_data = None

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').value,
            self.map_callback,
            10
        )

        # Publisher
        self.path_pub = self.create_publisher(
            Path,
            '/planned_path',
            10
        )

        # Action Server
        self._action_server = ActionServer(
            self,
            Navigation,
            'plan_path',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self.get_logger().info('Path Planner Node initialized')

    def get_pose_from_tf(self):
        """Get current robot pose from TF as a PoseStamped."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                Time(seconds=0),
                timeout=Duration(seconds=0.5),
            )
            pose = PoseStamped()
            pose.header = t.header
            pose.pose.position.x = t.transform.translation.x
            pose.pose.position.y = t.transform.translation.y
            pose.pose.position.z = t.transform.translation.z
            pose.pose.orientation = t.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None

    def map_callback(self, msg: OccupancyGrid):
        self.map_data = msg
        self.get_logger().info('Map received', once=True)

    def goal_callback(self, goal_request):
        
        self.get_logger().info('Received goal request')

        if self.get_pose_from_tf() is None:
            self.get_logger().warn('No current pose available (TF lookup failed), rejecting goal')
            return GoalResponse.REJECT

        if self.map_data is None:
            self.get_logger().warn('No map data available, rejecting goal')
            return GoalResponse.REJECT

        # Validate goal position
        goal_pose = goal_request.goal
        if not self.is_valid_goal(goal_pose):
            self.get_logger().warn('Invalid goal position, rejecting')
            return GoalResponse.REJECT

        self.get_logger().info('Goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Called when client requests to cancel the goal."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Main execution callback - computes the path using A*.
        """
        self.get_logger().info('Executing path planning...')

        # Get goal from request
        goal_pose = goal_handle.request.goal

        # Publish feedback
        feedback_msg = Navigation.Feedback()
        feedback_msg.feedback = 'Starting path planning'
        goal_handle.publish_feedback(feedback_msg)

        # ===============================================
        # YOUR A* IMPLEMENTATION GOES HERE
        # ===============================================
        try:
            # Get current pose from TF
            current_pose = self.get_pose_from_tf()
            if current_pose is None:
                self.get_logger().error('Cannot get robot pose from TF')
                goal_handle.abort()
                result = Navigation.Result()
                result.result = False
                return result

            # Convert poses to grid coordinates
            start_grid = self.world_to_grid(current_pose)
            goal_grid = self.world_to_grid(goal_pose)

            self.get_logger().info(f'Planning from {start_grid} to {goal_grid}')

            # Publish progress feedback
            feedback_msg.feedback = 'Computing A* path...'
            goal_handle.publish_feedback(feedback_msg)

          
            path_grid = self.astar_search(start_grid, goal_grid)

            if path_grid is not None:
                self.get_logger().info(f'Path found with {len(path_grid)} grid waypoints')

            if path_grid is None:
                self.get_logger().warn('No path found!')
                goal_handle.abort()
                result = Navigation.Result()
                result.result = False
                return result

            # Convert grid path back to world coordinates
            path_world = self.grid_path_to_world(path_grid)

            # Publish feedback
            feedback_msg.feedback = f'Path found with {len(path_world)} waypoints'
            goal_handle.publish_feedback(feedback_msg)

            # Publish path for visualization
            self.publish_path(path_world, goal_pose.header.frame_id)

            # Check if goal was cancelled during execution
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result = Navigation.Result()
                result.result = False
                return result

            # Success!
            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f'Path planning failed: {str(e)}')
            goal_handle.abort()
            result = Navigation.Result()
            result.result = False
            return result

        # Return result
        result = Navigation.Result()
        result.result = True
        self.get_logger().info('Path planning completed successfully')
        return result

    def astar_search(self, start, goal):
        """
        A* with Euclidean distance heuristic.
        Optimized: inlined neighbor/free checks, precomputed move costs, math.sqrt.
        """
        if self.map_data is None:
            self.get_logger().error('No map data available for A*')
            return None

        width = self.map_data.info.width
        height = self.map_data.info.height
        map_data = self.map_data.data

        goal_r, goal_c = goal

        # Precomputed 8-connected directions with move costs
        SQRT2 = 1.4142135623730951
        directions = (
            (-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, SQRT2),  (1, 0, 1.0),   (1, 1, SQRT2),
        )

        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))
        counter += 1

        came_from = {}
        g_score = {start: 0}
        closed_set = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            cr, cc = current
            cur_g = g_score[current]

            for dr, dc, cost in directions:
                nr = cr + dr
                nc = cc + dc

                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue

                if map_data[nr * width + nc] != 0:
                    continue

                nb = (nr, nc)
                if nb in closed_set:
                    continue

                tentative_g = cur_g + cost

                if nb not in g_score or tentative_g < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + sqrt((nr - goal_r)**2 + (nc - goal_c)**2)
                    heapq.heappush(open_set, (f, counter, nb))
                    counter += 1

        self.get_logger().warn('A* search failed: No path to goal')
        return None

    def reconstruct_path(self, came_from, current):
      
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()  # Path was built backwards
        return path

    def smooth_path(self, path):
        """
        Line-of-sight path smoothing.
        Skips intermediate waypoints when a straight line is obstacle-free.
        """
        if len(path) <= 2:
            return path

        width = self.map_data.info.width
        map_data = self.map_data.data

        smoothed = [path[0]]
        current = 0

        while current < len(path) - 1:
            # Try to skip as far ahead as possible
            farthest = current + 1
            for candidate in range(len(path) - 1, current + 1, -1):
                if self.line_of_sight(path[current], path[candidate], width, map_data):
                    farthest = candidate
                    break
            smoothed.append(path[farthest])
            current = farthest

        return smoothed

    def line_of_sight(self, p0, p1, width, map_data):
        """
        Bresenham line check — returns True if all cells on the line are free.
        """
        r0, c0 = p0
        r1, c1 = p1

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc

        r, c = r0, c0
        while True:
            if map_data[r * width + c] != 0:
                return False
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

        return True

    def is_valid_goal(self, goal_pose: PoseStamped) -> bool:
        if self.map_data is None:
            return False
        goal_grid = self.world_to_grid(goal_pose)
        row, col = goal_grid

        width = self.map_data.info.width
        height = self.map_data.info.height

        # Check if within map bounds
        if row < 0 or row >= height or col < 0 or col >= width:
            self.get_logger().warn(f'Goal {goal_grid} out of bounds')
            return False

       
        idx = row * width + col
        cell_value = self.map_data.data[idx]

        # 0 = free, 100 = occupied, -1 = unknown
        if cell_value != 0:
            self.get_logger().warn(f'Goal {goal_grid} not free (value={cell_value})')
            return False

        return True

    def world_to_grid(self, pose: PoseStamped) -> tuple:
        """Args:
            pose: PoseStamped in world frame

        Returns:
            (row, col) tuple in grid coordinates
        """
        if self.map_data is None:
            return (0, 0)

        # Get map parameters
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y

        # Convert to grid
        col = int((pose.pose.position.x - origin_x) / resolution)
        row = int((pose.pose.position.y - origin_y) / resolution)

        return (row, col)

    def grid_to_world(self, row: int, col: int) -> tuple:
        """
        Returns:
            (x, y) tuple in world frame (meters)
        """
        if self.map_data is None:
            return (0.0, 0.0)

        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y

        x = col * resolution + origin_x
        y = row * resolution + origin_y

        return (x, y)

    def grid_path_to_world(self, grid_path: list) -> list:
        """
        Args:
            grid_path: List of (row, col) tuples

        Returns:
            List of (x, y) tuples in world frame
        """
        world_path = []
        for row, col in grid_path:
            x, y = self.grid_to_world(row, col)
            world_path.append((x, y))
        return world_path

    def publish_path(self, path_world: list, frame_id: str):
        """
        Publish path for visualization in RViz.
        """
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in path_world:
            pose = PathPose()
            pose.header.frame_id = frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path with {len(path_world)} waypoints')


def main(args=None):
    rclpy.init(args=args)

    path_planner = PathPlannerNode()

    try:
        rclpy.spin(path_planner)
    except KeyboardInterrupt:
        pass
    finally:
        path_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
