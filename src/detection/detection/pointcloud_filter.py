#!/usr/bin/env python

import cv2
import numpy as np

import pprint
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from robp_interfaces.msg import ObjectDetectionMsg, ObjectDetectionArrayMsg


class PointcloudFilter(Node):

    CAMERA_HEIGHT = 0.087
    MAX_DEPTH = 1.5
    MAX_HEIGHT = 0.05 - CAMERA_HEIGHT
    DBSCAN_EPS = 0.015  # max distance between points in cluster
    DBSCAN_MIN_SAMPLES = 40  # minimum points to form a cluster
    TOO_BIG_NOT_TO_FAIL = 0.3  # Meaning it is bigger than even a box (SHOULD BE RENDERED USELESS BY STICKING WITH MAP LIMIT)
    YOU_THREE_MAKE_A_CUBE = 0.06 # verify if cube
    BIG_BEAUTIFUL_BOX_MIN = 0.13 # verify if box is big and beautiful Real size is ~15cm
    BIG_BEAUTIFUL_BOX_MAX = 0.30 # verify if box is big and beautiful 
    BIG_BEAUTIFUL_BOX_IS_LONG_SIDE = 0.175 # This is surely longer than the smaller side (real == ~24cm)
    BIG_BEAUTIFUL_BOX_BIG_ENOUGH_SPAN = 0.2

    CUBE_SIZE = 0.03 # size of the cube in meters

    def __init__(self):
        super().__init__('pointcloud_filter')

        self.get_logger().info(f'PointcloudFilter initialized with MAX_DEPTH={self.MAX_DEPTH} and MAX_HEIGHT={self.MAX_HEIGHT}')

        self._cloud_cb_group = ReentrantCallbackGroup()
        self._detection_cb_group = MutuallyExclusiveCallbackGroup()

        # TF for transforming detections to map frame
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pub = self.create_publisher(
            PointCloud2, '/filtered_points', 10)
        
        self._marker_pub = self.create_publisher(
            MarkerArray, '/cluster_centroids', 10)

        self._cluster_cloud_pub = self.create_publisher(
            PointCloud2, '/cluster_points', 10)

        self._detections_pub = self.create_publisher(
            ObjectDetectionArrayMsg, '/object_detections', 10)

        self.create_subscription(PointCloud2, '/realsense/depth/color/points', self.cloud_callback, 10, callback_group=self._cloud_cb_group)
        detection_qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/detection_on', self.detection_callback, detection_qos, callback_group=self._detection_cb_group)

        self.detection_on = True


        self._eliminated_pub = self.create_publisher(
            PointCloud2, '/eliminated', 10)

    # Hue bands for cube colors (OpenCV H scale 0-179)
    _HUE_RED_LOW = (0, 10)
    _HUE_RED_HIGH = (170, 180)
    _HUE_GREEN = (80, 88)
    _HUE_BLUE = (95, 110)
    _HUE_WOOD = (10, 20)
    _SAT_THRESHOLD = 0.2  # normalized; ignore points with S below this for hue classification
    _V_PERCENTILE_LOW = 10   # ignore darkest %
    _V_PERCENTILE_HIGH = 90  # ignore brightest %
    _S_PERCENTILE_LOW = 10   # ignore lowest % saturation for robust median (same trim as V for cubes)
    _S_PERCENTILE_HIGH = 90  # ignore highest % saturation for robust median
    _HUE_BOX_BLUEISH = (80, 100)  # OpenCV H (0–179); big-box branch if median hue in this band
    _MIN_COLOR_POINTS = 20   # winning color band must have at least this many points to qualify
    _MIN_CLUSTER_POINTS = 50  # cluster must have at least this many points to be classified at all
    _SAT_RB_MIN = 0.35  # minimum median saturation required for red/blue cubes
    _V_MIN = 0.10  # minimum median value (brightness) required to classify at all
    BOX_SLICE_HEIGHT_RATIO = 0.90
    BOX_LINE_RANSAC_ITERS = 120
    BOX_EDGE_MIN_LENGTH = 0.08
    BOX_LINE_INLIER_THRESHOLD = 0.01
    BOX_AXIS_PROBE_LENGTH = 0.10

    def _infer_cube_color_from_rgb(self, cluster_full: np.ndarray, debug_info: dict | None = None) -> str:
        """Infer cube color from RGB using HSV. Optionally fill debug_info with reason, counts, medians.
        Returns CUBE_R, CUBE_B, CUBE_G, CUBE_W, or CUBE_U."""


        if cluster_full.shape[0] == 0 or cluster_full.shape[1] < 4:
            return "CUBE_U"
        rgb_packed = cluster_full[:, 3].view(np.uint32)
        r = (rgb_packed >> 16) & 255
        g = (rgb_packed >> 8) & 255
        b = rgb_packed & 255
        rgb_uint8 = np.column_stack([r, g, b]).astype(np.uint8)
        rgb_for_cv2 = rgb_uint8.reshape(-1, 1, 3)
        hsv_points = cv2.cvtColor(rgb_for_cv2, cv2.COLOR_RGB2HSV).reshape(-1, 3)  # H [0,179], S,V [0,255]
        n_raw = hsv_points.shape[0]

        # Ignore darkest and brightest 10% by value
        v = hsv_points[:, 2].astype(np.float64)
        v_lo = np.percentile(v, self._V_PERCENTILE_LOW)
        v_hi = np.percentile(v, self._V_PERCENTILE_HIGH)
        mask_v = (v >= v_lo) & (v <= v_hi)
        hsv_filtered = hsv_points[mask_v]
        n_after_v = hsv_filtered.shape[0]
        if hsv_filtered.shape[0] == 0:
            return "CUBE_U"

        # Ignore low-saturation points before hue classification
        sat_threshold_uint8 = int(self._SAT_THRESHOLD * 255)
        s = hsv_filtered[:, 1]
        mask_s = s >= sat_threshold_uint8
        hsv_sat = hsv_filtered[mask_s]
        n_after_s = hsv_sat.shape[0]
        median_h_all = float(np.median(hsv_filtered[:, 0]))
        median_s_all = float(np.median(hsv_filtered[:, 1]) / 255.0)
        median_v_all = float(np.median(hsv_filtered[:, 2]) / 255.0)

        # Too dark overall -> do not classify (likely shadows / invalid color)
        if median_v_all < self._V_MIN:

            return "CUBE_U"

        if hsv_sat.shape[0] == 0:
            # All points low saturation: use median to decide wood vs unknown
            median_s = np.median(hsv_filtered[:, 1]) / 255.0
            median_h = np.median(hsv_filtered[:, 0])
            if median_s < self._SAT_THRESHOLD and (self._HUE_WOOD[0] <= median_h < self._HUE_WOOD[1]):
                return "CUBE_W"
            if self._HUE_WOOD[0] <= median_h < self._HUE_WOOD[1]:
                return "CUBE_W"
            return "CUBE_U"

        # Classify from hue histogram: count points in each color band
        h = hsv_sat[:, 0].astype(np.float64)
        mask_red = (h < self._HUE_RED_LOW[1]) | (h >= self._HUE_RED_HIGH[0])
        mask_green = (h >= self._HUE_GREEN[0]) & (h < self._HUE_GREEN[1])
        mask_blue = (h >= self._HUE_BLUE[0]) & (h < self._HUE_BLUE[1])
        mask_wood = (h >= self._HUE_WOOD[0]) & (h < self._HUE_WOOD[1])

        red_count = int(np.sum(mask_red))
        green_count = int(np.sum(mask_green))
        blue_count = int(np.sum(mask_blue))
        wood_count = int(np.sum(mask_wood))

        counts = [
            ("CUBE_R", red_count),
            ("CUBE_G", green_count),
            ("CUBE_B", blue_count),
            ("CUBE_W", wood_count),
        ]
        best_class, best_count = max(counts, key=lambda x: x[1])

        # Saturation stats computed only for points inside each band
        s_sat_norm = hsv_sat[:, 1].astype(np.float64) / 255.0
        sat_median_r = float(np.median(s_sat_norm[mask_red])) if red_count > 0 else None
        sat_median_g = float(np.median(s_sat_norm[mask_green])) if green_count > 0 else None
        sat_median_b = float(np.median(s_sat_norm[mask_blue])) if blue_count > 0 else None
        sat_median_w = float(np.median(s_sat_norm[mask_wood])) if wood_count > 0 else None
        sat_median_best = {
            "CUBE_R": sat_median_r,
            "CUBE_G": sat_median_g,
            "CUBE_B": sat_median_b,
            "CUBE_W": sat_median_w,
        }.get(best_class)

        if best_count < self._MIN_COLOR_POINTS:
            return "CUBE_U"

        if best_class in ("CUBE_B") and (sat_median_best is None or sat_median_best < self._SAT_RB_MIN):
            return "CUBE_U"
        return best_class

    def _publish_detection(self, centroid: np.ndarray, class_name: str, confidence: float,
                           frame_id: str, stamp, detections_list: list,
                           ground_axis: np.ndarray | None = None) -> ObjectDetectionMsg | None:
        """Transform centroid to map frame and append ObjectDetectionMsg to list.
        Uses the same stamp for TF lookup and outgoing message (must match received message stamp).
        """
        p = PointStamped()
        p.header.frame_id = frame_id
        p.header.stamp = stamp
        p.point.x = float(centroid[0])
        p.point.y = float(centroid[1])
        p.point.z = float(centroid[2])
        try:
            trans = self._tf_buffer.lookup_transform(
                'map', frame_id, stamp, timeout=Duration(seconds=0.1)
            )
            p_map = tf2_geometry_msgs.do_transform_point(p, trans)
            det = ObjectDetectionMsg()
            det.header.frame_id = 'map'
            det.header.stamp = stamp
            det.class_name = class_name
            det.pose.position = p_map.point
            det.pose.orientation.w = 1.0
            if ground_axis is not None:
                axis_norm = float(np.linalg.norm(ground_axis))
                if axis_norm > 1e-6:
                    axis_unit = ground_axis / axis_norm
                    axis_probe = PointStamped()
                    axis_probe.header.frame_id = frame_id
                    axis_probe.header.stamp = stamp
                    axis_probe.point.x = float(centroid[0] + axis_unit[0] * self.BOX_AXIS_PROBE_LENGTH)
                    axis_probe.point.y = float(centroid[1] + axis_unit[1] * self.BOX_AXIS_PROBE_LENGTH)
                    axis_probe.point.z = float(centroid[2] + axis_unit[2] * self.BOX_AXIS_PROBE_LENGTH)
                    axis_probe_map = tf2_geometry_msgs.do_transform_point(axis_probe, trans)
                    dx = float(axis_probe_map.point.x - p_map.point.x)
                    dy = float(axis_probe_map.point.y - p_map.point.y)
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                        yaw = float(np.arctan2(dy, dx))
                        # Yaw and yaw+pi represent the same edge direction.
                        # Canonicalize to the smallest absolute yaw in map frame
                        # so arrow orientation is stable across frames.
                        yaw_alt = float(np.arctan2(-dy, -dx))
                        if abs(yaw_alt) < abs(yaw):
                            yaw = yaw_alt
                        det.pose.orientation.z = float(np.sin(yaw * 0.5))
                        det.pose.orientation.w = float(np.cos(yaw * 0.5))
            det.confidence = confidence
            detections_list.append(det)
            return det
        except TransformException:
            return None

    def _append_detection_direction_marker(
        self,
        marker_array: MarkerArray,
        detection: ObjectDetectionMsg,
        marker_id: int,
    ) -> None:
        """Visualize detection orientation in map frame as an arrow."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = detection.header.stamp
        marker.ns = "box_directions"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = detection.pose
        marker.pose.position.y += 0.01
        marker.scale.x = 0.14
        marker.scale.y = 0.025
        marker.scale.z = 0.03
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = 1
        marker_array.markers.append(marker)

    def detection_callback(self, msg: Bool):
        self.detection_on = msg.data

    
    def _fit_dominant_line_ransac(self, ground_points):
        """Fit a dominant line to 2D slice points and return its direction and inliers."""
        n_points = ground_points.shape[0]
        if n_points < 4:
            return None

        rng = np.random.default_rng(0)
        best_inlier_mask = None
        best_direction = None
        best_span = 0.0
        best_score = -1.0

        for _ in range(self.BOX_LINE_RANSAC_ITERS):
            i0, i1 = rng.choice(n_points, size=2, replace=False)
            p0 = ground_points[i0]
            p1 = ground_points[i1]
            direction = p1 - p0
            norm = np.linalg.norm(direction)
            if norm < self.BOX_EDGE_MIN_LENGTH:
                continue

            direction /= norm
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            distances = np.abs((ground_points - p0) @ normal)
            inlier_mask = distances <= self.BOX_LINE_INLIER_THRESHOLD
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < 4:
                continue

            inlier_points = ground_points[inlier_mask]
            centered = inlier_points - np.mean(inlier_points, axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            refined_direction = vh[0]
            refined_norm = np.linalg.norm(refined_direction)
            if refined_norm < 1e-6:
                continue
            refined_direction /= refined_norm

            projections = centered @ refined_direction
            span = float(np.max(projections) - np.min(projections))
            score = float(inlier_count) * span

            if score > best_score:
                best_score = score
                best_inlier_mask = inlier_mask
                best_direction = refined_direction
                best_span = span

        if best_inlier_mask is None or best_direction is None or best_span < self.BOX_EDGE_MIN_LENGTH:
            return None

        return best_direction, best_inlier_mask, best_span

    def cloud_callback(self, msg: PointCloud2):
        # Use the received timestamp everywhere: TF lookup and all published messages
        if not self.detection_on:
            return

        received_stamp = msg.header.stamp
        data = pc2.read_points_numpy(msg, skip_nans=True)
        filtered = data[(data[:, 2] <= self.MAX_DEPTH) & (data[:, 1] >= -self.MAX_HEIGHT) & (data[:, 1] <= self.CAMERA_HEIGHT)]

        out_msg = pc2.create_cloud(msg.header, msg.fields, filtered)
        self._pub.publish(out_msg)

        # DBSCAN clustering
        if len(filtered) <= 0:
            return
        
        points = filtered[:, :3]
        labels = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MIN_SAMPLES).fit_predict(points)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]
        detections_list = []

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

            if cluster_full.shape[0] < self._MIN_CLUSTER_POINTS:
                continue

            cluster_min_x = np.min(cluster_points[:, 0], axis=0)
            cluster_max_x = np.max(cluster_points[:, 0], axis=0)
            cluster_size_x = np.max(cluster_max_x - cluster_min_x)

            # DEBUG: If cluster is too big for an object, publish it as eliminated 
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
            
            # TODO: CHECK THIS
            # FLOOR GANG (cough if its not on the floor, ignore)
            if cluster_min_y > self.CAMERA_HEIGHT + 0.01:
                # publish eliminated point cloud in cyan
                eliminated_rgb = np.full((cluster_points.shape[0], 3), [0.0, 1.0, 1.0], dtype=np.float32)
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

                # Check if its too TALL
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

                # Publish eliminated point cloud in Yellow (valid cube detection)
                debug_info = {}
                cube_class = self._infer_cube_color_from_rgb(cluster_full, debug_info)
                self.get_logger().info(f'Detected color: {cube_class}')
                centroid = np.median(cluster_points, axis=0)

                # push centroid 1.5 cm back in z (depth)
                centroid[2] = centroid[2] + self.CUBE_SIZE / 2
                self._publish_detection(centroid, cube_class, 0.7, msg.header.frame_id, received_stamp, detections_list)

                # publish in the cube color
                eliminated_rgb = np.full((cluster_points.shape[0], 3), [1.0, 1.0, 0.0], dtype=np.float32)
                r = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                g = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                b = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                packed = (r << 16) | (g << 8) | b
                rgb_float = packed.view(np.float32)
                cloud_with_rgb = np.column_stack([cluster_points, rgb_float])
                eliminated_msg = pc2.create_cloud(msg.header, msg.fields, cloud_with_rgb)
                self._eliminated_pub.publish(eliminated_msg)

            
            # Box detection 
            if self.BIG_BEAUTIFUL_BOX_MAX > cluster_size_x > self.BIG_BEAUTIFUL_BOX_MIN:
                # DEBUG: big-box path — show cluster on /eliminated in yellow
                eliminated_rgb = np.full((cluster_points.shape[0], 3), [1.0, 1.0, 0.0], dtype=np.float32)
                er = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                eg = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                eb = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                packed_dbg = (er << 16) | (eg << 8) | eb
                rgb_float_dbg = packed_dbg.view(np.float32)
                cloud_dbg = np.column_stack([cluster_points, rgb_float_dbg])
                eliminated_msg_dbg = pc2.create_cloud(msg.header, msg.fields, cloud_dbg)
                self._eliminated_pub.publish(eliminated_msg_dbg)

                # Get RGB from point cloud (x,y,z,rgb packed as float32)
                rgb_packed = cluster_full[:, 3].view(np.uint32)
                r = (rgb_packed >> 16) & 255
                g = (rgb_packed >> 8) & 255
                b = rgb_packed & 255
                rgb_uint8 = np.column_stack([r, g, b]).astype(np.uint8)
                rgb_for_cv2 = rgb_uint8.reshape(-1, 1, 3)
                hsv_points = cv2.cvtColor(rgb_for_cv2, cv2.COLOR_RGB2HSV).reshape(-1, 3)

                # HSV channels: H = [:,0], S = [:,1], V = [:,2]. S, V are in [0,255]
                s = hsv_points[:, 1].astype(np.float64)
                s_lo = np.percentile(s, self._S_PERCENTILE_LOW)
                s_hi = np.percentile(s, self._S_PERCENTILE_HIGH)
                mask_s_mid = (s >= s_lo) & (s <= s_hi)
                s_trimmed = s[mask_s_mid]
                if s_trimmed.size == 0:
                    median_saturation = float(np.median(s) / 255.0)
                else:
                    median_saturation = float(np.median(s_trimmed) / 255.0)
                h = hsv_points[:, 0].astype(np.float64)
                h_trimmed = h[mask_s_mid]
                if h_trimmed.size == 0:
                    median_hue = float(np.median(h))
                else:
                    median_hue = float(np.median(h_trimmed))
                max_saturation = float(np.max(s) / 255.0)
                min_saturation = float(np.min(s) / 255.0)
                intensity = float(np.mean(hsv_points[:, 2] / 255.0))
                blueish = self._HUE_BOX_BLUEISH[0] <= median_hue <= self._HUE_BOX_BLUEISH[1]
                woodish = self._HUE_WOOD[0] <= median_hue < self._HUE_WOOD[1]
                debug = (
                    f'Max saturation: {max_saturation} | Min saturation: {min_saturation} \n '+
                    f'Median saturation: {median_saturation} | Median hue: {median_hue} \n '+
                    f'Intensity: {intensity}'
                )

                if ((median_saturation < 0.40 and intensity < 0.42) or blueish) and not woodish:
                    
                    self.get_logger().info("BOX\n"+debug)
                    centroid = np.mean(cluster_points, axis=0)
                    eliminated_rgb = np.full((cluster_points.shape[0], 3), [1.0, 0.5, 0.0], dtype=np.float32)
                    r = (eliminated_rgb[:, 0] * 255).astype(np.uint32)
                    g = (eliminated_rgb[:, 1] * 255).astype(np.uint32)
                    b = (eliminated_rgb[:, 2] * 255).astype(np.uint32)
                    packed = (r << 16) | (g << 8) | b
                    rgb_float = packed.view(np.float32)
                    cloud_with_rgb = np.column_stack([cluster_points, rgb_float])
                    eliminated_msg = pc2.create_cloud(msg.header, msg.fields, cloud_with_rgb)
                    self._eliminated_pub.publish(eliminated_msg)


                    # Project onto a plane
                    min_y = float(np.min(cluster_points[:, 1]))
                    max_y = float(np.max(cluster_points[:, 1]))
                    object_height = max_y - min_y

                    target_y = max_y - self.BOX_SLICE_HEIGHT_RATIO * object_height
                    slice_half_thickness =  0.1 * object_height

                    slice_mask = np.abs(cluster_points[:, 1] - target_y) <= slice_half_thickness
                    slice_points = cluster_points[slice_mask]

                    plane_points = slice_points[:, [0, 2]]

                    line_fit1 = self._fit_dominant_line_ransac(plane_points)
                    if line_fit1 is None:
                        self.get_logger().warn("Ransac could not fit the first box edge")
                        self._publish_detection(centroid, "BOX", 0.7, msg.header.frame_id, received_stamp, detections_list)
                        continue

                    direction1, inlier_mask1, span1 = line_fit1
                    if span1 < self.BOX_EDGE_MIN_LENGTH:
                        self.get_logger().warn("Ransac first edge span too short")
                        self._publish_detection(centroid, "BOX", 0.7, msg.header.frame_id, received_stamp, detections_list)
                        continue

                    plane_points2 = plane_points[~inlier_mask1]
                    line_fit2 = self._fit_dominant_line_ransac(plane_points2)
                    if line_fit2 is None:
                        direction2 = None
                        inlier_mask2 = None
                        span2 = 0.0
                    else:
                        direction2, inlier_mask2, span2 = line_fit2

                    # Publish long edge axis
                    if direction2 is not None and span2 >= span1:
                        best_direction_2d = direction2
                        best_span = span2
                    else:
                        best_direction_2d = direction1
                        best_span = span1
                    
                    #self.get_logger().info(f"Best direction: {span1} {span2}")
                    # If the two are visible:
                    # Infer the box center from the corner of the box (intersection of the two lines)

                    if direction2 is not None and direction1 is not None:
                        line1_points = plane_points[inlier_mask1]
                        line2_points = plane_points2[inlier_mask2]
                        if line1_points.shape[0] >= 2 and line2_points.shape[0] >= 2:
                            p1 = np.mean(line1_points, axis=0)
                            p2 = np.mean(line2_points, axis=0)
                            A = np.column_stack([direction1, -direction2])
                            detA = np.linalg.det(A)

                            # Solve p1 + t*d1 = p2 + u*d2 in least-squares form.
                            if abs(detA) > 1e-6:
                                t_u = np.linalg.solve(A, p2 - p1)
                            else:
                                t_u, _, _, _ = np.linalg.lstsq(A, p2 - p1, rcond=None)
                            corner = p1 + t_u[0] * direction1

                            mean_plane = np.mean(plane_points, axis=0)
                            center_candidates = []
                            for sign1 in (-1.0, 1.0):
                                for sign2 in (-1.0, 1.0):
                                    candidate = (
                                        corner
                                        + 0.5 * sign1 * span1 * direction1
                                        + 0.5 * sign2 * span2 * direction2
                                    )
                                    center_candidates.append(candidate)
                            best_center_2d = min(
                                center_candidates,
                                key=lambda c: float(np.linalg.norm(c - mean_plane)),
                            )
                            centroid[0] = float(best_center_2d[0])
                            centroid[2] = float(best_center_2d[1])

                    # Rotate by 90 if its the short edge
                    if best_span < self.BIG_BEAUTIFUL_BOX_IS_LONG_SIDE:
                        rotated_direction_2d = np.array(
                            [-best_direction_2d[1], best_direction_2d[0]],
                            dtype=np.float64,
                        )
                        self.get_logger().info(
                            f"Rotating box yaw by 90 deg (observed span={best_span:.3f}m < "
                            f"long-side threshold={self.BIG_BEAUTIFUL_BOX_IS_LONG_SIDE:.3f}m)"
                        )
                        direction_for_yaw = rotated_direction_2d
                    else:
                        direction_for_yaw = best_direction_2d

                    box_ground_axis = np.array([direction_for_yaw[0], 0.0, direction_for_yaw[1]], dtype=np.float64)
                    box_detection = self._publish_detection(
                        centroid,
                        "BOX",
                        0.7,
                        msg.header.frame_id,
                        received_stamp,
                        detections_list,
                        ground_axis=box_ground_axis,
                    )
                    """
                    if box_detection is not None:
                        self._append_detection_direction_marker(
                            marker_array,
                            box_detection,
                            1000 + int(i),
                        )
                    """
                else:
                    # Box detection (high saturation
                    centroid = np.mean(cluster_points, axis=0)
                    #self._publish_detection(centroid, "BOX", 0.8, msg.header.frame_id, received_stamp, detections_list)
                    # publish eliminated point cloud in green
                    debug+=(f"is woodish: {woodish}, is blueish: {blueish}, is saturation: {median_saturation}")
                    
                    self.get_logger().warn("NOT BOX\n" +(debug))
               
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
            marker.header.stamp = received_stamp
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

        """
        if marker_array.markers:
            self._marker_pub.publish(marker_array)
        """

        # Publish object detections for mapper
        if detections_list:
            detections_msg = ObjectDetectionArrayMsg()
            detections_msg.header.frame_id = 'map'
            detections_msg.header.stamp = received_stamp
            detections_msg.detections = detections_list
            self._detections_pub.publish(detections_msg)

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