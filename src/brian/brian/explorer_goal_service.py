#!/usr/bin/env python

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from robp_interfaces.srv import GetGoal

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from tf_transformations import euler_from_quaternion

import numpy as np


class ExplorerGoalService(Node):


    def __init__(self):
        super().__init__("explorer")
        self.get_logger().info("Explorer node started, ready to spit out goals!")

        mutgroup = MutuallyExclusiveCallbackGroup()
        self._default_callback_group = ReentrantCallbackGroup() 

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            "/map/occupancy_grid",
            self.map_callback,
            10,
            callback_group=mutgroup
        )
        self.map = None

        self.goal_service = self.create_service(
            GetGoal,
            "explore_goal", 
            self.explore_request_callback, 
            callback_group=mutgroup
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)


    def map_callback(self, msg: OccupancyGrid):
        self.map = msg

    def explore_request_callback(self, request, response):
        self.get_logger().info("Recieved exploration goal request")
        if self.map == None:
            self.get_logger().warning("No map data available, returning an empty goal")
            return response
        try:
            (x,y,yaw) = self.get_explore_goal()
            response.x = x
            response.y = y
            response.yaw = yaw
        except TypeError as error:
            self.get_logger().warning(f"Oops, something went wrong: {error}")
        
        return response

    def find_frontiers(self, map_array):
        frontier_cells = []

        def is_frontier(i,j) -> bool:
            for k in range(i-1, i+2):
                for l in range(j-1, j+2):
                    if map_array[k][l] == -1:
                        return True
            return False

        h, w = map_array.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if map_array[i][j] == 0:
                    if is_frontier(i,j):
                        frontier_cells.append((i,j))
        return frontier_cells

    def get_explore_goal(self):
        if len(self.map.data) < 1:
            self.get_logger().error(f"Recieved and empty map")
            return
        map = np.reshape(self.map.data, (self.map.info.height, self.map.info.width))
        stamp = self.map.header.stamp
        from_frame_rel = self.map.header.frame_id
        to_frame_rel = "base_link"
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame=to_frame_rel,
                source_frame=from_frame_rel,
                time=Time().from_msg(stamp),
                timeout=Duration(seconds=1)
            )
        except TransformException:
            self.get_logger().error(f"Couldn't find transform from {from_frame_rel} to {to_frame_rel}")
            return
        
        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y
        q = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w
        ]
        robot_yaw = euler_from_quaternion(q)[2]

        # Map robot position to grid cell
        # Convert continuous (x, y) coordinates to grid indices using map resolution and origin
        grid_x = (robot_x - self.map.info.origin.position.x) / self.map.info.resolution
        grid_y = (robot_y - self.map.info.origin.position.y) / self.map.info.resolution
        # Clamp to map bounds
        grid_x = int(min(grid_x, self.map.info.width-1))
        grid_y = int(min(grid_y, self.map.info.height-1))

        frontiers = np.array(self.find_frontiers(map))

        # Find the closest unseen frontier cell (from the robot)
        # Calculate euclidean distance from robot position to each frontier cell
        rs = np.sum(([grid_x, grid_y] - frontiers)**2, axis=1)
        # Select frontier cell with minimum distance that's further away than 0.5 m
        mask = (rs > 0.5/self.map.info.resolution)
        rs = rs[mask]
        frontiers = frontiers[mask]
        if len(frontiers) == 0:
            self.get_logger().warning("No distant frontiers available, returning fallback goal")
            return robot_x, robot_y, robot_yaw
        (x,y) = frontiers[np.argmin(rs)]
        # Convert back from grid coordinates to world coordinates (x, y)
        x = x * self.map.info.resolution + self.map.info.origin.position.x
        y = y * self.map.info.resolution + self.map.info.origin.position.y
        # Calculate target yaw by computing atan2 angle from robot to frontier goal
        yaw = np.atan2(y - robot_y, x - robot_x) - robot_yaw
        yaw = yaw % (2 * np.pi)
        # TODO: If the selected frontier is outside the perimeter, they should be moved to the perimeter 
        # edge and we should also make sure the heading is perpendicular to the perimeter edge (facing out)
        self.get_logger().info(f"Sending goal at ({x=},{y=},{yaw=})")


        return x, y, yaw


def main():
    rclpy.init()
    node = ExplorerGoalService()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
