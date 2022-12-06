#!/usr/bin/env python

import roslib
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from tensorflow.keras import models
import os
from tensorflow import reshape
import numpy as np
import uuid

import time

from image_processing import process_image, process_crosswalk, process_pedestrian

LINEAR_SPEED = 1.743392200500000766e-01 * 1.5
ANGULAR_SPEED = 9.000000000000000222e-01 * 1.5

VID_LOCATION = "/home/rcritchlow/ENPH353_Team16_Data/"

SHAPE = (108, 192)

FPS = 30

class ImitationController:
    def __init__(self, timer):
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.timer = timer
        self.av_model = models.load_model('/home/rcritchlow/ros_ws/src/controller_package/nodes/rm5_modified_8.h5')
        self.crosswalk_turn_buffer = 0
        self.vel_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.twist_callback)

        self.twist = Twist()
        self.released = False

        self.current_image = []
        self.image_stream = []
        self.initial_crosswalk_image = []

        self.vel_data = np.empty((0,2))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.data_name = uuid.uuid4()
        self.video_writer = cv2.VideoWriter(f'{VID_LOCATION}{self.data_name}.mp4', fourcc, FPS, (SHAPE[1], SHAPE[0]), 0)

        self.pedestrian_scan = False
        self.pscan_count = 0
        self.scan_thresh = 3
    
    def twist_callback(self, data):
        self.twist = data

        if self.twist.linear.y != 0:
            movement = Twist()
            movement.linear.x = LINEAR_SPEED
            if self.twist.linear.y > 0:
                movement.angular.z = ANGULAR_SPEED
                self.video_writer.write(self.current_image)
                self.vel_data = np.append(self.vel_data, [[ANGULAR_SPEED, LINEAR_SPEED]], axis=0)
                self.vel_pub.publish(movement)
            if self.twist.linear.y < 0:
                movement.angular.z = -ANGULAR_SPEED
                self.video_writer.write(self.current_image)
                self.vel_data = np.append(self.vel_data, [[-1 * ANGULAR_SPEED, LINEAR_SPEED]], axis=0)
                self.vel_pub.publish(movement)

        if self.twist.linear.z != 0:
            self.video_writer.release()
            np.savetxt(f'/home/rcritchlow/ENPH353_Team16_Data/{self.data_name}.csv', self.vel_data, delimiter=',')
            print('released')

        return

    def image_callback(self, data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        processed_image = process_image(cv_image)
        self.current_image = processed_image

        movement = Twist()

        if self.crosswalk_turn_buffer <= 0:
            crosswalk_score = process_crosswalk(cv_image)
            if crosswalk_score >= 400:
                movement.linear.x = 0
                movement.angular.z = 0
                self.vel_pub.publish(movement)
                if self.pedestrian_scan == False:
                    print('logged image')
                    self.pedestrian_scan = True
                    self.crosswalk_turn_buffer = 20
                    self.initial_crosswalk_image = cv_image
        else: 
            crosswalk_score = 0
        
        if self.pedestrian_scan:
            pedestrian_score = process_pedestrian(self.initial_crosswalk_image, cv_image)
            if pedestrian_score > self.scan_thresh and pedestrian_score < 20:
                self.pedestrian_scan = False
                self.scan_thresh = 10
                self.pscan_count = 0
            else: 
                if self.pscan_count >= 50:
            
                    self.scan_thresh = 1
                    self.initial_crosswalk_image = cv_image
                self.pscan_count += 1

                return

        cv2.imshow('stream', cv_image)
        cv2.waitKey(3)

        if self.twist.linear.y != 0:
            return

        prediction = self.av_model.predict(reshape(processed_image, (1, 108, 192, 1)), verbose=0)[0]
        i = np.argmax(prediction)

        movement.linear.x = LINEAR_SPEED

        # print(prediction)
        # print(i)

        if i == 0:
            movement.angular.z = 0
        elif i == 1:
            movement.angular.z = ANGULAR_SPEED
            self.crosswalk_turn_buffer -= 1
        elif i == 2:
            movement.angular.z = -1 * ANGULAR_SPEED
            self.crosswalk_turn_buffer -= 1
        else:
            movement.angular.z = 0
      
        # print(f'turn buffer: {self.crosswalk_turn_buffer}')

        try:
            self.vel_pub.publish(movement)
        except CvBridgeError as e:
            print(e)

        return
