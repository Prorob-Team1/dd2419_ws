#!/usr/bin/env python

import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray


class PointcloudFilter(Node):

    CAMERA_HEIGHT = 0.087
    MAX_DEPTH = 1.5
    MAX_HEIGHT = 0.05 - CAMERA_HEIGHT
    DBSCAN_EPS = 0.015  # max distance between points in cluster
    DBSCAN_MIN_SAMPLES = 40  # minimum points to form a cluster
    TOO_BIG_NOT_TO_FAIL = 0.3  # Meaning it is bigger than even a box (SHOULD BE RENDERED USELESS BY STICKING WITH MAP LIMIT)
    YOU_THREE_MAKE_A_CUBE = 0.06 # verify if cube
    BIG_BEAUTIFUL_BOX_potential = 0.12 # verify if box is big and beautiful

    def __init__(self):
        super().__init__('pointcloud_filter')

        self.get_logger().info(f'PointcloudFilter initialized with MAX_DEPTH={self.MAX_DEPTH} and MAX_HEIGHT={self.MAX_HEIGHT}')

        self._pub = self.create_publisher(
            PointCloud2, '/filtered_points', 10)
        
        self._marker_pub = self.create_publisher(
            MarkerArray, '/cluster_centroids', 10)

        self._cluster_cloud_pub = self.create_publisher(
            PointCloud2, '/cluster_points', 10)

        self.create_subscription(
            PointCloud2, '/realsense/depth/color/points', self.cloud_callback, 10)

        self._eliminated_pub = self.create_publisher(
            PointCloud2, '/eliminated', 10)

    def cloud_callback(self, msg: PointCloud2):
        data = pc2.read_points_numpy(msg, skip_nans=True)
        filtered = data[(data[:, 2] <= self.MAX_DEPTH) & (data[:, 1] >= -self.MAX_HEIGHT) & (data[:, 1] <= self.CAMERA_HEIGHT)]

        out_msg = pc2.create_cloud(msg.header, msg.fields, filtered)
        self._pub.publish(out_msg)

        # DBSCAN clustering
        if len(filtered) > 0:
            points = filtered[:, :3]
            dbscan_start = time.perf_counter()
            labels = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MIN_SAMPLES).fit_predict(points)
            dbscan_time = time.perf_counter() - dbscan_start

            labelling_start = time.perf_counter()
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels >= 0]

            colors = [
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
                (1.0, 0.5, 0.0),
                (0.5, 0.0, 1.0),
            ]

            marker_array = MarkerArray()
            for i, cluster_id in enumerate(unique_labels):
                cluster_mask = labels == cluster_id
                cluster_full = filtered[cluster_mask]
                cluster_points = cluster_full[:, :3]
                cluster_min_x = np.min(cluster_points[:, 0], axis=0)
                cluster_max_x = np.max(cluster_points[:, 0], axis=0)
                cluster_size_x = np.max(cluster_max_x - cluster_min_x)
                if cluster_size_x > self.TOO_BIG_NOT_TO_FAIL:
                    # publish eliminated point cloud in red
                    eliminated_rgb = np.full((cluster_points.shape[0], 3), [1.0, 0.0, 0.0], dtype=np.float32)
                    r = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                    g = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                    b = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                    packed = (r << 16) | (g << 8) | b
                    rgb_float = packed.view(np.float32)
                    cloud_with_rgb = np.column_stack([cluster_points, rgb_float])
                    eliminated_msg = pc2.create_cloud(msg.header, msg.fields, cloud_with_rgb)
                    self._eliminated_pub.publish(eliminated_msg)
                    continue  # Do not publish marker for this cluster
                
                cluster_min_y = np.min(cluster_points[:, 1], axis=0)
                cluster_max_y = np.max(cluster_points[:, 1], axis=0)

                # FLOOR GANG
                if cluster_min_y > self.CAMERA_HEIGHT + 0.01:
                    # publish eliminated point cloud in red
                    eliminated_rgb = np.full((cluster_points.shape[0], 3), [1.0, 0.0, 0.0], dtype=np.float32)
                    r = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                    g = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                    b = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                    packed = (r << 16) | (g << 8) | b
                    rgb_float = packed.view(np.float32)
                    cloud_with_rgb = np.column_stack([cluster_points, rgb_float])
                    eliminated_msg = pc2.create_cloud(msg.header, msg.fields, cloud_with_rgb)
                    self._eliminated_pub.publish(eliminated_msg)
                    continue
                
                # cube detection
                if cluster_size_x < self.YOU_THREE_MAKE_A_CUBE:
                    #if min above floor and max below ceiling cntinue
                    if cluster_min_y < -0.03 + self.CAMERA_HEIGHT:
                        # publish eliminated point cloud in red
                        eliminated_rgb = np.full((cluster_points.shape[0], 3), [1.0, 0.0, 0.0], dtype=np.float32)
                        r = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                        g = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                        b = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                        packed = (r << 16) | (g << 8) | b
                        rgb_float = packed.view(np.float32)
                        cloud_with_rgb = np.column_stack([cluster_points, rgb_float])
                        eliminated_msg = pc2.create_cloud(msg.header, msg.fields, cloud_with_rgb)
                        self._eliminated_pub.publish(eliminated_msg)
                        continue

                    # Publish eliminated point cloud in Yellow
                    eliminated_rgb = np.full((cluster_points.shape[0], 3), [1.0, 1.0, 0.0], dtype=np.float32)
                    r = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                    g = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                    b = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                    packed = (r << 16) | (g << 8) | b
                    rgb_float = packed.view(np.float32)
                    cloud_with_rgb = np.column_stack([cluster_points, rgb_float])
                    eliminated_msg = pc2.create_cloud(msg.header, msg.fields, cloud_with_rgb)
                    self._eliminated_pub.publish(eliminated_msg)
                    continue  # Do not publish marker for this cluster


                # check HSV values to verify if box is big and beautiful (aka it should be bigger than 0 yet lower than 18-22, novinhas moment)
                if cluster_size_x > self.BIG_BEAUTIFUL_BOX_potential:
                    # Get RGB from point cloud (x,y,z,rgb packed as float32)

                    rgb_packed = cluster_full[:, 3].view(np.uint32)
                    r = (rgb_packed >> 16) & 255
                    g = (rgb_packed >> 8) & 255
                    b = rgb_packed & 255
                    rgb_uint8 = np.column_stack([r, g, b]).astype(np.uint8)
                    rgb_for_cv2 = rgb_uint8.reshape(-1, 1, 3)
                    hsv_points = cv2.cvtColor(rgb_for_cv2, cv2.COLOR_RGB2HSV).reshape(-1, 3)

                    # HSV channels: H = [:,0], S = [:,1], V = [:,2]. S, V are in [0,255]
                    average_saturation = np.mean(hsv_points[:, 1] / 255.0)
                    max_saturation = np.max(hsv_points[:, 1] / 255.0)
                    min_saturation = np.min(hsv_points[:, 1] / 255.0)
                    mean_saturation = np.mean(hsv_points[:, 1] / 255.0)
                    intensity = np.mean(hsv_points[:, 2] / 255.0)
                    self.get_logger().info(f'Max saturation: {max_saturation} | Min saturation: {min_saturation} | Mean saturation: {mean_saturation} | Intensity: {intensity}')

                    if mean_saturation < 0.25:
                        eliminated_rgb = np.full((cluster_points.shape[0], 3), [1.0, 0.5, 0.0], dtype=np.float32)
                        r = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                        g = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                        b = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                        packed = (r << 16) | (g << 8) | b
                        rgb_float = packed.view(np.float32)
                        cloud_with_rgb = np.column_stack([cluster_points, rgb_float])
                        eliminated_msg = pc2.create_cloud(msg.header, msg.fields, cloud_with_rgb)
                        self._eliminated_pub.publish(eliminated_msg)
                        continue
                    else:
                        # publish eliminated point cloud in green
                        eliminated_rgb = np.full((cluster_points.shape[0], 3), [0.0, 1.0, 0.0], dtype=np.float32)
                        r = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                        g = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                        b = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                        packed = (r << 16) | (g << 8) | b
                        rgb_float = packed.view(np.float32)
                        cloud_with_rgb = np.column_stack([cluster_points, rgb_float])
                        eliminated_msg = pc2.create_cloud(msg.header, msg.fields, cloud_with_rgb)
                        self._eliminated_pub.publish(eliminated_msg)
                        continue

                centroid = np.mean(cluster_points, axis=0)
                marker = Marker()
                marker.header.frame_id = msg.header.frame_id
                marker.header.stamp = msg.header.stamp
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.id = int(i)
                marker.pose.position.x = float(centroid[0])
                marker.pose.position.y = float(centroid[1])
                marker.pose.position.z = float(centroid[2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                color = colors[i % len(colors)]
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 1.0
                marker.lifetime.sec = 1
                marker_array.markers.append(marker)

            if marker_array.markers:
                self._marker_pub.publish(marker_array)

            # Publish clusters as a single point cloud (exclude noise points)
            mask = labels >= 0
            if mask.any():
                points_cluster = points[mask]
                labels_cluster = labels[mask]
                rgb_01 = np.array([colors[l % len(colors)] for l in labels_cluster], dtype=np.float32)
                r = (rgb_01[:, 0] * 255).astype(np.uint32)
                g = (rgb_01[:, 1] * 255).astype(np.uint32)
                b = (rgb_01[:, 2] * 255).astype(np.uint32)
                packed = (r << 16) | (g << 8) | b
                rgb_float = packed.view(np.float32)
                cluster_cloud = np.column_stack([points_cluster, rgb_float])
                cluster_msg = pc2.create_cloud(msg.header, msg.fields, cluster_cloud)
                self._cluster_cloud_pub.publish(cluster_msg)
            labelling_time = time.perf_counter() - labelling_start
            self.get_logger().info(f'DBSCAN: {dbscan_time*1000:.2f} ms | Total labelling: {labelling_time*1000:.2f} ms | ')


def main():
    rclpy.init()
    node = PointcloudFilter()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
