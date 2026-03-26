#!/usr/bin/env python

from email.mime import image
import math

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, Imu
from rclpy.time import Time

from collections import deque
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from pathlib import Path


class BoxDetector(Node):

    def __init__(self):
        super().__init__("box_detector")

        self.image_sub = self.create_subscription(
            Image, "/arm/camera/image_raw", self.image_callback, 10
        )
        self.processed_image_pub = self.create_publisher(
            Image, "/arm/camera/img_debug", 10
        )
        pattern_dir = Path("./patterns")
        pattern = sorted(pattern_dir.glob("*.png"))[0]
        self.pattern_img = cv2.imread(str(pattern), cv2.IMREAD_GRAYSCALE)
        self.bridge = CvBridge()

    def image_callback(self, msg: Image):
        # threshold the image
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 20
        )

        # dilate by 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.dilate(thresh, kernel, iterations=1)

        templates = [
            self.pattern_img,
            cv2.rotate(self.pattern_img, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(self.pattern_img, cv2.ROTATE_180),
            cv2.rotate(self.pattern_img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

        all_matches = []
        boxes = []
        scores = []
        shapes = []
        threshold = 0.7
        for template in templates:

            # do template matching to find the pattern in the image
            res = cv2.matchTemplate(closed, template, cv2.TM_CCOEFF_NORMED)

            # keep original max print (unchanged)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # print(f"Max correlation: {max_val:.4f} at {max_loc}")

            # =========================
            # ADD: multi-match detection
            # =========================

            locations = np.where(res >= threshold)
            matches = list(zip(*locations[::-1]))

            # print(f"Matches above threshold {threshold}: {len(matches)}")

            h, w = template.shape

            # # OPTIONAL: Non-Maximum Suppression (to remove overlaps)

            for pt in matches:
                boxes.append([pt[0], pt[1], w, h])
                scores.append(res[pt[1], pt[0]])
                shapes.append((w, h))

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, score_threshold=threshold, nms_threshold=0.3
            )

            for i in indices:
                i = i[0] if isinstance(i, (list, tuple)) else i
                x, y, w, h = boxes[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # publish the processed image
        processed_msg = Image()
        processed_msg.header = msg.header
        processed_msg.height = img.shape[0]
        processed_msg.width = img.shape[1]
        processed_msg.encoding = "bgr8"
        processed_msg.data = img.tobytes()
        self.processed_image_pub.publish(processed_msg)


def main():
    rclpy.init()
    node = BoxDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
        # rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    executor.shutdown()
