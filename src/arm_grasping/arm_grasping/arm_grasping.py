#!/usr/bin/env python3
# coding=utf8
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from robp_interfaces.msg import ArmControl 
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
import time
import threading
from std_srvs.srv import Trigger
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# ==========================================
# PID Controller Class
# ==========================================
class PID:
    def __init__(self, P=0.2, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()

    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.windup_guard = 20.0 
        self.output = 0.0

    def update(self, feedback_value):
        error = self.SetPoint - feedback_value
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            self.last_time = self.current_time
            self.last_error = error
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
            self.output = - self.output

    def setWindup(self, windup):
        self.windup_guard = windup

# ==========================================
# Inverse Kinematics Solver
# ==========================================
class ArmIK:
    def __init__(self):
        self.L1, self.L2, self.L3 = 0.065, 0.101, 0.094
        self.L4 = 0.116 

    def solve(self, u_x, u_y, u_z, alpha=-75):
        x, y, z = u_z, u_y, u_x
        try:
            theta_base = math.atan2(y, x)
            r = math.sqrt(x*x + y*y)
            z_t = z - self.L1
            alpha_rad = math.radians(alpha)
            m = r - self.L4 * math.cos(alpha_rad)
            n = z_t - self.L4 * math.sin(alpha_rad)
            L_reach = math.sqrt(m*m + n*n)
            if L_reach > (self.L2 + self.L3 + 0.005): return None
            cos_el = (self.L2**2 + self.L3**2 - L_reach**2) / (2 * self.L2 * self.L3)
            theta_el = math.acos(max(min(cos_el, 1.0), -1.0))
            beta = math.atan2(n, m)
            psi = math.acos(max(min((self.L2**2 + L_reach**2 - self.L3**2) / (2 * self.L2 * L_reach), 1.0), -1.0))
            return [math.degrees(theta_base), math.degrees(beta + psi), 180.0 - math.degrees(theta_el), (math.degrees(beta + psi) - (180.0 - math.degrees(theta_el)) - alpha)]
        except: return None

# ==========================================
# Vision-Based Smart Tracking Node
# ==========================================
class ArmGraspingServer(Node):
    def __init__(self):
        super().__init__('ArmGraspingServer_node')
        
        # === Target Center ===
        self.TARGET_CENTER_X = 320   
        self.TARGET_CENTER_Y = 450  
        
        # === PID params ===
        # Y-axis (lateral) Left-Right
        self.pid_lateral = PID(P=0.00001, I=0.0000, D=0.000001)
        # Z-axis (depth)  Forward-Backward
        self.pid_depth   = PID(P=0.00001, I=0.0000, D=0.000001)
        
        self.pid_lateral.SetPoint = self.TARGET_CENTER_X
        self.pid_depth.SetPoint   = self.TARGET_CENTER_Y

        # Movement constraints
        self.MAX_STEP = 0.005      
        self.DEADZONE = 3        

        # State management
        self.stable_count = 0
        self.STABLE_LIMIT = 18
        self.lost_target_count = 0
        
        # Current arm position 
        self.init_pos = [0.06, 0.0, 0.12] 
        self.curr_pos = [0.06, 0.0, 0.12] 
        self.prepare_pos = [0.06, 0.0, 0.14]
        self.hold_pos = [0.14, 0.0, 0.12] 
        self.drop_pos = [0.08, 0.0, 0.22]
        self.Y_LIMIT = 0.15 
        self.Z_MIN = 0.12
        self.Z_MAX = 0.22

        self.state = "IDLE"

        # Publishers and Subscribers
        self.control_pub = self.create_publisher(ArmControl, '/arm/control', 10)
        self.debug_img_pub = self.create_publisher(Image, 'arm/camera/img_debug', 10)
        self._last_debug_publish = 0.0
        self.create_subscription(Image, 'arm/camera/image_raw', self.image_callback, 10)
        self.CallbackGroup = MutuallyExclusiveCallbackGroup()
        self.grasp_srv = self.create_service(Trigger,'Start_Grasping',self.grasp_service_callback, callback_group = self.CallbackGroup)
        self.drop_srv = self.create_service(Trigger,'Start_Dropping',self.drop_service_callback, callback_group = self.CallbackGroup)

        self.grasp_success = False
        self.grasp_Event = threading.Event()

        self.dropping_success = False
        self.dropping_Event = threading.Event()
        
        self.ik = ArmIK()
        self.bridge = CvBridge()
        self.cv_image = None
        
        
        self.TEST_COLOR = 'blue'

        # HSV Color Ranges

        self.HSV_RANGES = { 
            'red': [
                ((0, 140, 100), (2, 255, 255)),       
                ((175, 140, 100), (180, 255, 255)),
                ((165, 92, 80), (180, 255, 255)),
                ((0, 92, 80), (6, 255, 255))
            ],
            
            'green': [
                ((50, 80, 46), (85, 255, 255)),
                ((75, 100, 80), (98, 255, 255)),       
            ],
            
            'blue': [
                ((100, 80, 46), (124, 255, 255)),
                ((80, 20, 190), (150, 255, 255)),       
            ]
        }
        
        threading.Thread(target=self.main_loop, daemon=True).start()
        self.get_logger().info("Arm Grasping Service is ready....")

    def grasp_service_callback(self,request,response):

        self.get_logger().info("Received grasp request!")

        if self.state != "IDLE":
            response.success = False
            response.message = f"Arm is currently busy (State: {self.state})."
            return response
        
        self.grasp_Event.clear()
        self.grasp_success = False
        self.state = "SEARCHING"
        self.lost_target_count = 0

        self.get_logger().info("Executing task... waiting for result...")

        finished_in_time = self.grasp_Event.wait(timeout=60.0)
        
        
        if not finished_in_time:
            self.state = "IDLE" 
            response.success = False
            response.message = "Timeout: Failed to grasp within 60 seconds."
            self.get_logger().warn(response.message)
        else:
            response.success = self.grasp_success
            if self.grasp_success:
                response.message = "Successfully picked up the object!"
                self.get_logger().info(response.message)
            else:
                response.message = "Failed to find or grab the object."
                self.get_logger().warn(response.message)
                
        return response
        
    def drop_service_callback(self,request,response):

        self.get_logger().info("Received dropping request!")

        if self.state != "HOLDING":
            response.success = False
            response.message = f"Arm is not holding the item (State: {self.state})."
            return response
        
        self.dropping_Event.clear()
        self.dropping_success = False
        self.state = "DROPPING"
        

        self.get_logger().info("Executing task... waiting for result...")

        finished_in_time = self.dropping_Event.wait(timeout=20.0)

        if not finished_in_time:
            self.state = "HOLDING" 
            response.success = False
            response.message = "Timeout: Failed to drop within 20 seconds."
            self.get_logger().warn(response.message)
        else:
            response.success = self.dropping_success
            if self.dropping_success:
                response.message = "Successfully dropped the object!"
                self.get_logger().info(response.message)
            else:
                response.message = "Failed to drop the object."
                self.get_logger().warn(response.message)
                
        return response
        
        
      




    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def send_arm_cmd(self, x, y, z, gripper_angle=20.0, claw_angle = 0.0, time_ms=100):
        
        # Pick IK solution with a fixed claw angle 60 degrees (pointing downwards)
        res = self.ik.solve(x, y, z, alpha=-60)
        #if not res: res = self.ik.solve(x, y, z, alpha=-75)
            
        if res:
            base, sh, el, wr = res
            msg = ArmControl()
            msg.header.stamp = self.get_clock().now().to_msg()
            hw = [0.0] * 6
            hw[0] = float(gripper_angle)
            hw[1] = 120.0 + claw_angle
            hw[2] = 120.0 - wr    
            hw[3] = 120.0 + el    
            hw[4] = 30.0 + sh     
            hw[5] = 120.0 + base  
            msg.position = hw
            msg.time = [int(time_ms)] * 6 
            self.control_pub.publish(msg)
            return True
        return False
    
    def send_dropoff_arm_cmd(self, gripper_angle, base, time_ms=1500):
        max_base_angle = 50
        if abs(base) > max_base_angle:
            self.get_logger().error("Base angle too large, clipping")
            base = np.clip(base, -max_base_angle, max_base_angle)
        msg = ArmControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        hw = [0.0] * 6
        hw[0] = gripper_angle
        hw[1] = 120.0
        hw[2] = 24.522165
        hw[3] = 120.0   
        hw[4] = 65.47784
        hw[5] = 120.0 + base  
        msg.position = hw
        msg.time = [int(time_ms)] * 6 
        self.control_pub.publish(msg)
    
    
    def execute_grasp(self,angle):
        self.get_logger().info(">>> STARTING GRASP SEQUENCE <<<")
        
        # Optimized: Z-Offset Compensation for better grasping reliability
        OFFSET_Z = 0.01476
        OFFSET_Y = 0.0
        
        # 1. Lower arm 
        # Note: If your calibration is perfect, you might not need OFFSET_Z
        # But usually, it helps to pull back slightly when going down.
        GRASP_HEIGHT = -0.0402
        target_z = self.curr_pos[2] + OFFSET_Z
        target_z = max(target_z, self.Z_MIN) # Safety

        if abs(angle)>0:
            OFFSET_Z += 0.01

        self.get_logger().info(f"1. Going Down to {GRASP_HEIGHT}...")
        self.send_arm_cmd(GRASP_HEIGHT, self.curr_pos[1] , target_z, 20.0, angle, time_ms=1500)
        time.sleep(2.0) 
        
        # 2. Close Gripper
        self.get_logger().info("2. Closing Gripper...")
        self.send_arm_cmd(GRASP_HEIGHT, self.curr_pos[1] , target_z, 108.0, angle, time_ms=1000)
        time.sleep(1.5)
        
        # 3. Lift Up
        self.get_logger().info("3. Lifting Up...")
        self.send_arm_cmd(self.hold_pos[0], self.hold_pos[1], self.hold_pos[2], 108.0, 0.0, time_ms=2000)
        time.sleep(2.5)
        
        self.get_logger().info(">>> GRASP COMPLETE <<<")
        self.state = "HOLDING"

        self.grasp_success = True
        self.grasp_Event.set()


    def execute_drop(self, box_center):
        if box_center is None:
            box_center = 0

        base_angle = np.clip(-box_center * 1.5 * 50, -50, 50)
        self.get_logger().info(f"Start Dropping at angle {base_angle}...")

        #self.send_arm_cmd(self.drop_pos[0], self.drop_pos[1], self.drop_pos[2], 108.0, 0.0, time_ms=2000, log=True)
        self.send_dropoff_arm_cmd(gripper_angle=108.0, base=base_angle, time_ms=2000)
        time.sleep(2.0)

        # self.send_arm_cmd(self.drop_pos[0], self.drop_pos[1], self.drop_pos[2], 30.0, 0.0, time_ms=2000)
        self.send_dropoff_arm_cmd(gripper_angle=30.0, base=base_angle, time_ms=2000)
        time.sleep(2.0)

        self.send_arm_cmd(self.init_pos[0], self.init_pos[1], self.init_pos[2], 30.0, 0.0, time_ms=2000)
        time.sleep(2.0)

        self.curr_pos = list(self.init_pos)
        
        # === FIX: Clear PID memory for the next grasp task ===
        self.pid_lateral.clear()
        self.pid_depth.clear()
        self.pid_lateral.SetPoint = self.TARGET_CENTER_X
        self.pid_depth.SetPoint = self.TARGET_CENTER_Y
        
        self.get_logger().info(">>> DROP COMPLETE <<<")
        self.state = "IDLE"

        self.dropping_success = True
        self.dropping_Event.set()

    def process_vision(self):
        
        if self.cv_image is None: return None
        
        img = self.cv_image.copy()
        
        # Preprocessing: Convert image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # === 1. Create a list to store all valid candidate blocks found ===
        valid_objects = []
        
        # Iterate through all target colors defined in the dictionary
        for target_color, ranges in self.HSV_RANGES.items():
            mask = None
            for (lower, upper) in ranges:
                curr = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = curr if mask is None else cv2.bitwise_or(mask, curr)
                
            # Morphological operations to clean up noise (erode then dilate)
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            
            # Find contours on the cleaned mask
            cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if cnts:
                # Find the largest contour by area for the current color
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                
                if area > 500:
                    rect = cv2.minAreaRect(c)
                    raw_cx = int(rect[0][0])
                    raw_cy = int(rect[0][1])
                    angle = rect[2]

                    # Angle correction to align the gripper with the longest edge
                    w, h = rect[1]
                    if w < h:
                        angle += 90
                    while angle > 45: angle -= 90
                    while angle < -45: angle += 90
                    
                    # === 2. Calculate distance: Euclidean distance from the block to TARGET_CENTER ===
                    dist = math.hypot(raw_cx - self.TARGET_CENTER_X, raw_cy - self.TARGET_CENTER_Y)
                    
                    # Append the current block's information to the list as a dictionary
                    valid_objects.append({
                        'color': target_color,
                        'cx': raw_cx,
                        'cy': raw_cy,
                        'angle': angle,
                        'dist': dist,
                        'rect': rect
                    })

        # === 3. Sorting and Selection ===
        if valid_objects:
            # Sort the valid objects list in ascending order based on 'dist' (distance)
            valid_objects.sort(key=lambda x: x['dist'])
            
            # Extract the first item (the closest target to the center)
            best_obj = valid_objects[0]
            
            # Visualization (draw boxes and markers only for the optimal target)
            box = cv2.boxPoints(best_obj['rect'])
            box = np.int64(box)
            cv2.drawContours(img, [box], -1, (0, 255, 255), 2)
            
            cv2.drawMarker(img, (self.TARGET_CENTER_X, self.TARGET_CENTER_Y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(img, (best_obj['cx'], best_obj['cy']), 6, (0, 255, 0), -1)
            
            # Calculate errors for PID control
            err_x = best_obj['cx'] - self.TARGET_CENTER_X
            err_y = best_obj['cy'] - self.TARGET_CENTER_Y
            
            # Display tracking data and currently locked color/distance on the screen
            cv2.putText(img, f"Err: {err_x}, {err_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f"Target: {best_obj['color'].upper()} Dist: {int(best_obj['dist'])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            self._publish_debug_image(img)
            #cv2.imshow("Smart Vision", img)
            #cv2.waitKey(1)
            
            # Finally, return only the coordinates and errors of the closest target
            return (best_obj['cx'], best_obj['cy'], best_obj['angle'], err_x, err_y)
        
        # === 4. Fallback: If no valid objects were found in the entire frame ===
        cv2.drawMarker(img, (self.TARGET_CENTER_X, self.TARGET_CENTER_Y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, "Target: NONE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        #cv2.imshow("Smart Vision", img)
        #cv2.waitKey(1)

        self._publish_debug_image(img)
        
        return None
    
    def process_vision_dropoff(self):
        if self.cv_image is None: return None
        
        img = self.cv_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 20
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        h, w = thresh.shape
        cv2.rectangle(thresh, (0,h*3//4), (w,h), (0,0), -1)

        # compute the variance for each column
        col_var = np.var(thresh, axis=0)

        # reshape to (1, N) so OpenCV treats it as an image row
        col_var_2d = col_var.reshape(1, -1)

        # apply Gaussian blur
        col_var_smooth = cv2.GaussianBlur(col_var_2d, (3, 1), 0)

        # flatten back to 1D
        col_var_smooth = col_var_smooth.flatten()
        x = np.arange(col_var_smooth.shape[0])
        # center of mass column
        var_sum = np.sum(col_var_smooth)
        if abs(var_sum) < 1e-5:
            self.get_logger().warn("No features detected for drop-off.")
            return None
        col_com = np.sum(col_var_smooth * x) / var_sum

        cv2.line(
            img,
            (int(col_com), 0),
            (int(col_com), img.shape[0]),
            (0, 255, 0),
            2,
        )
        relative_box_col_com = np.clip(((col_com / w) - 0.5) * 2, -1, 1)
        self._publish_debug_image(img)
        return relative_box_col_com


    def _publish_debug_image(self, img):
        now = time.time()
        if now - self._last_debug_publish < 0.1:  # 10 fps cap
            return
        self._last_debug_publish = now
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.debug_img_pub.publish(msg)
    

    def main_loop(self):
        time.sleep(2)
        self.send_arm_cmd(self.curr_pos[0], self.curr_pos[1], self.curr_pos[2], 20.0, 0, time_ms=3000)
        time.sleep(3.5)

        while rclpy.ok():

            if self.state == "IDLE" or self.state == "HOLDING":
                time.sleep(0.5)
                continue

            if self.state == "DROPPING":
                box_center = self.process_vision_dropoff()
                self.execute_drop(box_center)
                continue


            vision_result = self.process_vision()
            
            if vision_result:
                cx, cy, angle, e_x ,e_y = vision_result

                OFFSET_Y = 0.0

                if abs(angle)<20:
                    angle = 0

                #if angle>0:
                #    OFFSET_Y = 0.01

                # Reset lost counter because we see the object
                self.lost_target_count = 0
                
                # --- State 1: Search and Control ---

                if abs(e_x) > self.DEADZONE or abs(e_y) > self.DEADZONE:
                    self.stable_count = 0 
                    self.state = "TRACKING"
                
                    # === PID Calculation ===
                    self.pid_lateral.update(cx)
                    self.pid_depth.update(cy)
                    
                    # PID Output Interpretation: Step size for Y and Z axes
                    # Note: We negate the PID output because if the object is to the right (cx > TARGET_CENTER_X), we want to move left (negative Y direction), and if it's below (cy > TARGET_CENTER_Y), we want to move back (negative Z direction).
                    dx = -self.pid_lateral.output
                    dy = -self.pid_depth.output 
                    
                    # Deadzone check
                    if abs(cx - self.TARGET_CENTER_X) < self.DEADZONE: dx = 0
                    if abs(cy - self.TARGET_CENTER_Y) < self.DEADZONE: dy = 0
                    
                    # Limit step size to prevent overshooting and ensure smooth movement
                    step_y = np.clip(dx, -self.MAX_STEP, self.MAX_STEP)
                    step_z = np.clip(dy, -self.MAX_STEP, self.MAX_STEP)
                    
                    # Send command only if there's a significant movement needed (outside of deadzone)
                    if step_y != 0 or step_z != 0:
                        pred_y = np.clip(self.curr_pos[1] + step_y, -self.Y_LIMIT, self.Y_LIMIT)
                        pred_z = np.clip(self.curr_pos[2] + step_z, self.Z_MIN, self.Z_MAX)
                        
                        if self.send_arm_cmd(self.curr_pos[0], pred_y, pred_z, 0.0, 0, time_ms=30):
                            self.curr_pos[1] = pred_y
                            self.curr_pos[2] = pred_z

                # --- State 2: STABLE ---
                else:
                    self.stable_count += 1
                    self.state = "STABLE"
                    print(f"Stabilizing... {self.stable_count}/{self.STABLE_LIMIT}")
                    print(f"Claw angle... {angle}")
                    
                    if self.stable_count > self.STABLE_LIMIT:
                        self.state = "GRASPING"
                        self.send_arm_cmd(self.curr_pos[0], self.curr_pos[1], self.curr_pos[2], 20.0, angle, time_ms=500)
                        time.sleep(1.5)
                        self.execute_grasp(angle)

            # === IF Object NOT Detected ===
            else:
                self.lost_target_count += 1
                
                # If object lost for > 30 frames (approx 1 second), return Preparing Position
                if self.lost_target_count > 30:
                    
                    print("Target Lost... Returning Preparing Position.")
                    self.send_arm_cmd(self.prepare_pos[0], self.prepare_pos[1], self.prepare_pos[2], 20.0, 0, time_ms=1000)
                    time.sleep(1.5)

                    self.curr_pos = list(self.prepare_pos)
                          
                    # Reset PID to prevent "jump" when object is found again
                    self.pid_lateral.clear()
                    self.pid_depth.clear()
                    self.pid_lateral.SetPoint = self.TARGET_CENTER_X
                    self.pid_depth.SetPoint = self.TARGET_CENTER_Y
     
            time.sleep(0.05) 

def main():

    rclpy.init()
    node =  ArmGraspingServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()