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

COL_CROP_RATIO = 5/8
ROW_RATIO = 6/8

class pid_controller:
    
    def __init__(self):
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        rows = cv_image.shape[0]
        cols = cv_image.shape[1]

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0,0,125], dtype=np.uint8)
        upper_white = np.array([0,0,255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        res = cv2.bitwise_and(cv_image,cv_image, mask= mask)
        
        res_v = cv2.split(res)[2]

        crop = res_v[int(ROW_RATIO*rows):rows, int(cols*COL_CROP_RATIO):cols]
        blur = cv2.blur(crop,(9,9))
        # _, binary = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
        

        contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cx = 0
        cy = 0

        move = Twist()
        m = 10
        move.linear.x = 0.3
        # move.angular.z = 0

        if len(contours) > 0:
            cv2.drawContours(cv_image, contours, -1, (0,0,255), 3)
            cnt = contours[0]
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

            cx_center = int(cx+cols*COL_CROP_RATIO/2)
            # print(f"{cx}, {cols*COL_CROP_RATIO}, {cols/2}, {cx+cols*COL_CROP_RATIO/2-200}")
            print(f"cx_center: {cx_center}")
            print(1-2*(cx_center)/cols)
            move.angular.z = (1-2*(cx_center)/cols)*m
            cv2.circle(cv_image, (cx_center, cy+int(rows*ROW_RATIO)), 15, (0, 255, 255), 2)
            cv2.circle(crop, (cx, cy), 15, (0, 255, 255), 2)

        # cv2.imshow('img', bgr)
        cv2.imshow('res',res)
        cv2.imshow('img', cv_image)
        cv2.imshow("crop",crop)
        # cv2.imshow('binary',binary)
        cv2.waitKey(3)

        try:
            self.vel_pub.publish(move)
        except CvBridgeError as e:
            print(e)


