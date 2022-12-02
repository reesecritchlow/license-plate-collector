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

LOWER_WHITE = np.array([0,0,86], dtype=np.uint8)
UPPER_WHITE = np.array([127,17,206], dtype=np.uint8)
# LOWER_WHITE = np.array([96,0,70], dtype=np.uint8)
# UPPER_WHITE = np.array([125,82,200], dtype=np.uint8)

COL_CROP_RATIO = 5/8
ROW_RATIO = 3/8
MIN_AREA = 9000
MAX_AREA = 30000

WIDTH = 350
HEIGHT = 500
PERSPECTIVE_OUT = np.float32([[0,0], [0,HEIGHT-1], [WIDTH-1,HEIGHT-1], [WIDTH-1,0]])


class license_detector:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)
        

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    
        rows = cv_image.shape[0]
        cols = cv_image.shape[1]

        processed_img = self.process_image(cv_image, rows, cols)
        cv_image =  cv_image[int(2/5*rows):int(4/5*rows), 0:cols]
       

        contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)
        print(f"AREA: {cv2.contourArea(c)}, type {type(c)}")

        matrix = None

        # if we see the front section on a car
        if len(contours) > 0 and cv2.contourArea(c) < MAX_AREA and cv2.contourArea(c) > MIN_AREA:
            cv2.drawContours(cv_image, [c], 0, (0,0,255), 3)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, peri*0.05, True)[0:4]

            # print(approx)
            # approx = self.corner_fix(approx)
            perspective_in = np.float32(approx) 
 
            matrix = cv2.getPerspectiveTransform(perspective_in,PERSPECTIVE_OUT)

            cv2.drawContours(cv_image, [approx], 0, (0, 255, 0), 3)
            imgOutput = cv2.warpPerspective(cv_image, matrix, (WIDTH,HEIGHT), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

            license_plate = imgOutput[350:435, 15:WIDTH-15]
            parking_spot = imgOutput[120:340, 15:WIDTH-15]

            numbers = (license_plate[:,10:70], license_plate[:,70:130], license_plate[:,190:250], license_plate[:,250:310])
            numbers_img = np.concatenate((numbers[0], numbers[1], numbers[2], numbers[3]), axis=1)
            
            cv2.imshow('numbers', numbers_img)
            cv2.imshow('spot', parking_spot) 
            # cv2.imshow('plate', license_plate)

        cv2.imshow('image', cv_image)
        # cv2.imshow('blur',crop)
        cv2.waitKey(3)
        # This method should just get the image and call other functions
    
    def corner_fix(self, contour, tolerance = 10):
        
        """
        Ensures that contour corners are located arranged properly. 
        Top left, bottom left, bottom right, top right.

        @param contour must only have 4 values.

        Args:
            contour: 3D numpy array representing polygon contour

        Returns:
            3D numpy array representing a polygon contour, with indeces: top left, bottom left, bottom right, top right order.

        BAD\n
        tl: [[296  79]], bl: [[131  78]], br: [[129 222]], tr: [[293 199]]\n
        GOOD\n
        tl: [[238  78]], bl: [[239 203]], br: [[362 186]], tr: [[361  77]]
        """
        tl, bl, br, tr = (corner for corner in contour)
        if abs(tl[0,0] - bl[0,0]) > tolerance and abs(tr[0,0] - br[0,0]) > tolerance:
            if abs(tl[0, 1] - bl[0,1]) < tolerance and abs(br[0, 1] - bl[0,1]) > tolerance:
                return np.array([contour[1], contour[2], contour[3], contour[0]])            
        
        return np.array([tl, bl, br, tr])

    def process_image(self, image, rows, cols):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
        res = cv2.bitwise_and(image,image, mask= mask)

        split = cv2.split(res)[2]
        blur = cv2.blur(split,(5,5))
        thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
        
        crop = thresh[int(2/5*rows):int(4/5*rows), 0:cols]
        
        return crop
    