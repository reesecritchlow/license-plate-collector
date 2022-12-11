#!/usr/bin/env python

import roslib
# roslib.load_manifest('enph353_ros_lab')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from timer_controller import TimerController

COL_CROP_RATIO = 5/8
ROW_RATIO = 6/8
TIMER_END = 12

UPPER_WHITE = np.array([0,0,255], dtype=np.uint8)
LOWER_WHITE = np.array([0,0,125], dtype=np.uint8)

X_VEL = 0.3
TURN_RATIO = 10

class pid_controller:
    
    def __init__(self, timer):
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)
        self.timer_count = 0
        
        self.timer = timer

    def process_image(self, image, rows, cols):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
        res = cv2.bitwise_and(image,image, mask= mask)
        
        res_v = cv2.split(res)[2]

        crop = res_v[int(ROW_RATIO*rows):rows, int(cols*COL_CROP_RATIO):cols]
        blur = cv2.blur(crop,(9,9))

        return blur

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        rows = cv_image.shape[0]
        cols = cv_image.shape[1]

        processed_img = self.process_image(cv_image, rows, cols)        

        contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cx = 0
        cy = 0

        move = Twist()
        move.linear.x = X_VEL

        if len(contours) > 0:
            cv2.drawContours(cv_image, contours, -1, (0,0,255), 3)
            cnt = contours[0]
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

            cx_center = int(cx+cols*COL_CROP_RATIO/2)
            # print(f"{cx}, {cols*COL_CROP_RATIO}, {cols/2}, {cx+cols*COL_CROP_RATIO/2-200}")
            # print(f"cx_center: {cx_center}")
            # print(1-2*(cx_center)/cols)

            move.angular.z = (1-2*(cx_center)/cols)*TURN_RATIO
            cv2.circle(cv_image, (cx_center, cy+int(rows*ROW_RATIO)), 15, (0, 255, 255), 2)
        else:
            self.timer_count += 1

        if self.timer_count > TIMER_END:
            self.timer.terminate()
            rospy.signal_shutdown('pid lost line')

        cv2.imshow('img', cv_image)
        cv2.waitKey(3)

        try:
            self.vel_pub.publish(move)
        except CvBridgeError as e:
            print(e)




