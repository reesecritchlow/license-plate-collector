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

UPPER_WHITE = np.array([0,0,255*0.8], dtype=np.uint8)
LOWER_WHITE = np.array([0,0,255*0.4], dtype=np.uint8)

COL_CROP_RATIO = 5/8
ROW_RATIO = 6/8

class license_detector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        processed_img = self.process_image(cv_image)
        processed_img = cv2.split(processed_img)[2]
        contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image, contours, -1, (0,0,255), 3)
        
        cv2.imshow('img',cv_image)
        cv2.imshow('processed',processed_img)
        cv2.waitKey(3)
        # This method should just get the image and call other functions
    
    def process_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
        res = cv2.bitwise_and(image,image, mask= mask)
        
        return res
    