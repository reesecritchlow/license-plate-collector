#!/usr/bin/env python

import roslib
# roslib.load_manifest('enph353_ros_lab')
import sys
import os
from scipy.spatial import distance as dist 
import rospy
import cv2
import imutils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import pickle


LOWER_WHITE = np.array([0,0,86], dtype=np.uint8)
UPPER_WHITE = np.array([127,17,206], dtype=np.uint8)
# LOWER_WHITE = np.array([96,0,70], dtype=np.uint8)
# UPPER_WHITE = np.array([125,82,200], dtype=np.uint8)

COL_CROP_RATIO = 5/8
ROW_RATIO = 3/8
MIN_AREA = 10_000
MAX_AREA = 28_000
MIN_PLATE_AREA = 8_000
MAX_PLATE_AREA = 30_000


WIDTH = 600
HEIGHT = 1200
PERSPECTIVE_OUT = np.float32([[0,0], [0,HEIGHT-1], [WIDTH-1,HEIGHT-1], [WIDTH-1,0]])


import pickle
d = {'a':0,'b':1,'c':2}
with open('sample.txt', 'wb') as f:
    pickle.dump(d,f)


class license_detector:

    def __init__(self, plate_number, collect_data = False):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)
        self.plate_save = False
        self.plate_num = int(plate_number)
        self.collect_data = collect_data
        self.max_area = 0
        
        
        

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
            
        matrix = None

        # if we see the front section on a car
        # print(f"front: {len(contours) > 0 and cv2.contourArea(c) < MAX_AREA and cv2.contourArea(c) > MIN_AREA}")
        if len(contours) > 0:
            c = max(contours, key = cv2.contourArea)
            
            if (cv2.contourArea(c) < MAX_AREA 
                and cv2.contourArea(c) > MIN_AREA 
                and cv2.contourArea(c)):

                cv2.drawContours(cv_image, [c], 0, (0,0,255), 3)
                # find, and draw approximate polygon for contour c
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, peri*0.05, True)[0:4]
                
                if cv2.contourArea(c) > self.max_area and len(approx) == 4:
                    # furthest_pt = max(np.where([])) 
                    corner = max([(sum(pt[0]), i) for i, pt in enumerate(c)])
                    corner_coords = np.array([c[corner[1]][0,1], c[corner[1]][0,0]])

                    self.max_area = cv2.contourArea(c)

                    perspective_in = self.corner_fix(approx)
                    cv2.drawContours(cv_image, [approx], 0, (0, 255, 0), 3)

                    # matrix transformation for perpective shift of license plate
                    matrix = cv2.getPerspectiveTransform(perspective_in,PERSPECTIVE_OUT)
                    imgOutput = cv2.warpPerspective(cv_image, matrix, (WIDTH,HEIGHT), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

                    # cropping sections
                    license_plate = imgOutput[int(HEIGHT*0.7):int(HEIGHT*0.87), 15:WIDTH-15]
                    numbers = (license_plate[:,int(WIDTH*0.03):int(WIDTH*0.2)], license_plate[:,int(WIDTH*0.2):int(WIDTH*0.4)], license_plate[:,int(WIDTH*0.56):int(WIDTH*0.714)], license_plate[:,int(WIDTH*0.714):int(WIDTH*0.886)])
                    parking_spot = imgOutput[int(HEIGHT*0.24):int(HEIGHT*0.68), 15:WIDTH-15]

                    # displaying all 
                    # numbers_img = np.concatenate((numbers[0], numbers[1], numbers[2], numbers[3]), axis=1)
                    # numbers_img_post = self.contour_format(numbers_img, threshold=30)

                    plate_post = self.contour_format(license_plate)

                    number_cnt, _ = cv2.findContours(plate_post, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    total_num_area = np.sum([cv2.contourArea(cnt) for cnt in number_cnt])
                    print(f"NUMBER AREA:{total_num_area}")
                    
                    if (len(number_cnt) > 0 
                        and MIN_PLATE_AREA < total_num_area < MAX_PLATE_AREA  
                        and corner_coords[0] != cv_image.shape[0]-1 
                        and corner_coords[1] != cv_image.shape[1]-1):

                        if self.collect_data:
                            print("SAVE")
                            self.plate_save = True
                            cv2.imwrite(f"/home/fizzer/data/images/post/plate_post{self.plate_num}.png", plate_post)
                            cv2.imwrite(f"/home/fizzer/data/images/plate/plate{self.plate_num}.png", license_plate)
                            cv2.imwrite(f"/home/fizzer/data/images/parking/parking{self.plate_num}.png", parking_spot)
                            
                        cv2.imshow('numbers_POST', plate_post)
                        cv2.imshow('numbers', license_plate)
                    else:
                        if self.plate_save:
                            self.plate_save = False
                            self.plate_num += 1

                        
                    cv2.drawContours(license_plate, number_cnt, -1, (0, 255, 0), 1)
                    

                    parking_spot_post = self.contour_format(parking_spot)

                    # cv2.imshow('spot', parking_spot_post)
            else:
                self.max_area = 0

        cv2.imshow('image', cv_image)
        cv2.waitKey(3)
        # This method should just get the image and call other functions
    
    def corner_fix(self, contour, tolerance = 10):
        """
        Orders contour points in contour clockwise direction, starting from the top left corner.
        https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/ 

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
        pts = np.array([pt[0] for pt in contour])

        sort_x = pts[np.argsort(pts[:, 0]), :]

        leftMost = sort_x[:2, :]
        rightMost = sort_x[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        return np.array([[tl, bl, br, tr]], dtype="float32")

    def contour_format(self, image, blur_factor = 7, threshold = 10, lower = np.array([0, 0, 0]), upper=np.array([144, 85, 255])):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        split = cv2.split(result)[2]
        blur = cv2.blur(split,(blur_factor,blur_factor))
        thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)[1]

        return thresh

    def process_image(self, image, rows, cols):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
        res = cv2.bitwise_and(image,image, mask= mask)

        split = cv2.split(res)[2]
        blur = cv2.blur(split,(5,5))
        thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
        
        crop = thresh[int(2/5*rows):int(4/5*rows), 0:cols]
        
        return crop
    