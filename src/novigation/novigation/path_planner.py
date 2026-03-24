import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros import TransformListener, Buffer

from robp_interfaces.action import Navigation
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped as PathPose
from std_msgs.msg import Empty

from math import sqrt
import heapq


class PathPlannerNode(Node):

    def __init__(self):
        super().__init__('path_planner')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('planning_timeout', 5.0)

        self.goal_tolerance = 0.05

        self._active_goal_handle = None
        self._goal_lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_data = None

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.get_parameter('map_topic').value,
            self.map_callback,
            10
        )

        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.tail_pub = self.create_publisher(Path, '/tail_path', 10)
        self.cancel_pub = self.create_publisher(Empty, '/cancel_navigation', 10)

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

    def _clear_active_handle(self, goal_handle):
        with self._goal_lock:
            if self._active_goal_handle is goal_handle:
                self._active_goal_handle = None

    def get_pose_from_tf(self):
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
            self.get_logger().warn('No current pose available, rejecting goal')
            return GoalResponse.REJECT

        if self.map_data is None:
            self.get_logger().warn('No map data available, rejecting goal')
            return GoalResponse.REJECT

        goal_pose = goal_request.goal
        if not self.is_valid_goal(goal_pose):
            self.get_logger().warn('Invalid goal position, rejecting')
            return GoalResponse.REJECT

        self.get_logger().info('Goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        with self._goal_lock:
            if self._active_goal_handle is not None and self._active_goal_handle.is_active:
                self.get_logger().info('Preempting previous goal')
                self._active_goal_handle.abort()
            self._active_goal_handle = goal_handle

        self.get_logger().info('Executing path planning...')

        goal_pose = goal_handle.request.goal
        
        feedback_msg = Navigation.Feedback()

        try:
            current_pose = self.get_pose_from_tf()
            if current_pose is None:
                self.get_logger().error('Cannot get robot pose from TF')
                goal_handle.abort()
                return self._make_result(False)

            start_grid = self.world_to_grid(current_pose)
            goal_grid = self.world_to_grid(goal_pose)

            # Use a straight-line approach tail when orientation is explicitly set
            q = goal_pose.pose.orientation
            use_tail = (abs(q.w - 1.0) > 1e-4 or abs(q.x) > 1e-4
                        or abs(q.y) > 1e-4 or abs(q.z) > 1e-4)

            if use_tail:
                # x-axis of goal frame in world = approach direction
                approach_dx = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                approach_dy = 2.0 * (q.x * q.y + q.w * q.z)

                # Snap along the approach axis 
                snap_ref = PoseStamped()
                snap_ref.header = goal_pose.header
                snap_ref.pose.position.x = goal_pose.pose.position.x - 10.0 * approach_dx
                snap_ref.pose.position.y = goal_pose.pose.position.y - 10.0 * approach_dy
                snap_ref_grid = self.world_to_grid(snap_ref)
                snapped = self.find_nearest_free_cell(goal_grid, snap_ref_grid)
            else:
                snapped = self.find_nearest_free_cell(goal_grid, start_grid)

            if snapped is None:
                self.get_logger().warn('No free cell found near goal')
                goal_handle.abort()
                return self._make_result(False)

            snapped_wx, snapped_wy = self.grid_to_world(snapped[0], snapped[1])
            goal_in_obstacle = (snapped != goal_grid)
            if goal_in_obstacle:
                self.get_logger().info(f'Goal in obstacle, snapped from {goal_grid} to {snapped}')

            if use_tail:
                tail_length = 0.5
                pre_goal = PoseStamped()
                pre_goal.header = goal_pose.header
                pre_goal.pose.position.x = snapped_wx - tail_length * approach_dx
                pre_goal.pose.position.y = snapped_wy - tail_length * approach_dy
                pre_goal.pose.orientation.w = 1.0
                pre_goal_grid = self.world_to_grid(pre_goal)
                pre_snapped = self.find_nearest_free_cell(pre_goal_grid, start_grid)
                if pre_snapped is None:
                    pre_snapped = snapped
                self.get_logger().info(
                    f'Tail approach: planning to ({pre_goal.pose.position.x:.2f},'
                    f' {pre_goal.pose.position.y:.2f}), tail to ({snapped_wx:.2f}, {snapped_wy:.2f})'
                )
                plan_grid = pre_snapped
            else:
                plan_grid = snapped

            self.get_logger().info(f'Planning from {start_grid} to {plan_grid}')
            path_grid = self.astar_search(start_grid, plan_grid)
            if path_grid is None:
                self.get_logger().warn('No path found')
                goal_handle.abort()
                return self._make_result(False)

            self.get_logger().info(f'Path found with {len(path_grid)} grid cells')
            path_world = self.grid_path_to_world(path_grid)

            self.publish_path(path_world, goal_pose.header.frame_id)

            if use_tail:
                last_x, last_y = path_world[-1]
                res = self.map_data.info.resolution
                dist = sqrt((snapped_wx - last_x) ** 2 + (snapped_wy - last_y) ** 2)
                n_steps = max(2, int(dist / res))
                tail_world = [(last_x + (i / n_steps) * (snapped_wx - last_x),
                               last_y + (i / n_steps) * (snapped_wy - last_y))
                              for i in range(1, n_steps + 1)]
                self.publish_path(tail_world, goal_pose.header.frame_id, pub=self.tail_pub)
            else:
                self.publish_path([], goal_pose.header.frame_id, pub=self.tail_pub)

            feedback_msg.feedback = 'Navigating...'
            goal_handle.publish_feedback(feedback_msg)

            # Track to snapped free cell
            goal_x, goal_y = snapped_wx, snapped_wy
            rate = self.create_rate(10)

            while rclpy.ok():
                if not goal_handle.is_active:
                    self.get_logger().info('Goal preempted by newer goal')
                    return self._make_result(False)

                if goal_handle.is_cancel_requested:
                    self.cancel_pub.publish(Empty())
                    goal_handle.canceled()
                    self._clear_active_handle(goal_handle)
                    self.get_logger().info('Goal cancelled')
                    return self._make_result(False)

                current = self.get_pose_from_tf()
                if current is not None:
                    dx = current.pose.position.x - goal_x
                    dy = current.pose.position.y - goal_y
                    dist = sqrt(dx * dx + dy * dy)
                    feedback_msg.feedback = f'dist_to_goal={dist:.2f}'
                    goal_handle.publish_feedback(feedback_msg)
                    if dist < self.goal_tolerance:
                        self.cancel_pub.publish(Empty())  
                        goal_handle.succeed()
                        self._clear_active_handle(goal_handle)
                        self.get_logger().info('Goal reached')
                        return self._make_result(True)

                rate.sleep()

        except Exception as e:
            self.get_logger().error(f'Path planning failed: {e}')
            if goal_handle.is_active:
                goal_handle.abort()
            self._clear_active_handle(goal_handle)
            return self._make_result(False)

        return self._make_result(False)

    def _make_result(self, success):
        result = Navigation.Result()
        result.result = success
        return result

    def astar_search(self, start, goal):
        if self.map_data is None:
            return None

        width = self.map_data.info.width
        height = self.map_data.info.height
        map_data = self.map_data.data

        goal_r, goal_c = goal

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
                return self._reconstruct_path(came_from, current)

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

                cell_cost = map_data[nr * width + nc]
                if cell_cost == 100:
                    continue

                nb = (nr, nc)
                if nb in closed_set:
                    continue

                tentative_g = cur_g + cost + cell_cost * 0.05

                if nb not in g_score or tentative_g < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + sqrt((nr - goal_r)**2 + (nc - goal_c)**2)
                    heapq.heappush(open_set, (f, counter, nb))
                    counter += 1

        self.get_logger().warn('A* found no path to goal')
        return None

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def is_valid_goal(self, goal_pose: PoseStamped) -> bool:
        if self.map_data is None:
            return False
        row, col = self.world_to_grid(goal_pose)
        width = self.map_data.info.width
        height = self.map_data.info.height
        if row < 0 or row >= height or col < 0 or col >= width:
            self.get_logger().warn(f'Goal ({row}, {col}) out of bounds')
            return False
        return True

    def find_nearest_free_cell(self, goal, start):
        width = self.map_data.info.width
        height = self.map_data.info.height
        map_data = self.map_data.data

        if map_data[goal[0] * width + goal[1]] != 100:
            return goal

        gr, gc = goal
        sr, sc = start
        dr = sr - gr
        dc = sc - gc
        steps = max(abs(dr), abs(dc))

        for i in range(1, steps + 1):
            nr = gr + round(dr * i / steps)
            nc = gc + round(dc * i / steps)
            if 0 <= nr < height and 0 <= nc < width:
                if map_data[nr * width + nc] != 100:
                    return (nr, nc)

        return None

    def world_to_grid(self, pose: PoseStamped) -> tuple:
        if self.map_data is None:
            return (0, 0)
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        col = int((pose.pose.position.x - origin_x) / resolution)
        row = int((pose.pose.position.y - origin_y) / resolution)
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> tuple:
        if self.map_data is None:
            return (0.0, 0.0)
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        x = col * resolution + origin_x
        y = row * resolution + origin_y
        return (x, y)

    def grid_path_to_world(self, grid_path: list) -> list:
        return [self.grid_to_world(r, c) for r, c in grid_path]

    def publish_path(self, path_world: list, frame_id: str, pub=None):
        if pub is None:
            pub = self.path_pub
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

        pub.publish(path_msg)
        self.get_logger().info(f'Published path with {len(path_world)} waypoints')


def main(args=None):
    rclpy.init(args=args)

    path_planner = PathPlannerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(path_planner)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        path_planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()