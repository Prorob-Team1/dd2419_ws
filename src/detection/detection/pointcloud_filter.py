#!/usr/bin/env python

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
    MAX_DEPTH = 2.5
    MAX_HEIGHT = 0.15 - CAMERA_HEIGHT
    DBSCAN_EPS = 0.01  # max distance between points in cluster
    DBSCAN_MIN_SAMPLES = 40  # minimum points to form a cluster

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

    def cloud_callback(self, msg: PointCloud2):
        data = pc2.read_points_numpy(msg, skip_nans=True)
        filtered = data[(data[:, 2] <= self.MAX_DEPTH) & (data[:, 1] >= -self.MAX_HEIGHT) & (data[:, 1] <= self.CAMERA_HEIGHT)]

        out_msg = pc2.create_cloud(msg.header, msg.fields, filtered)
        self._pub.publish(out_msg)

        # DBSCAN clustering
        if len(filtered) > 0:
            points = filtered[:, :3]
            labels = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MIN_SAMPLES).fit_predict(points)

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
                cluster_points = points[labels == cluster_id]
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
