#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from tensorflow.keras import models
from tensorflow import reshape
import numpy as np
import uuid

from functions.image_processing import process_road, process_crosswalk, process_pedestrian

LINEAR_SPEED = 1.743392200500000766e-01 * 1.5
ANGULAR_SPEED = 9.000000000000000222e-01 * 1.5

ROAD_IMAGE_SHAPE = (108, 192)

CROSSWALK_STOP_THRESH = 400
CROSSWALK_TURN_BUFFER = 20  # number of turn actions to pass before looking for another crosswalk

SECOND_PEDESTRIAN_COUNT_THRESH = 50  # number of pedestrian samples before resampling
SECOND_LOWER_PEDESTRIAN_THRESH = 1
SECOND_UPPER_PEDESTRIAN_THRESH = 10000


class OutsideController:
    def __init__(self, timer):
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.timer = timer
        self.av_model = models.load_model('/home/rcritchlow/ros_ws/src/controller_package/nodes/rm5_modified_10.h5')

        self.current_road_image = []
        self.image_stream = []
        self.first_crosswalk_image = []

        self.pedestrian_scan = False
        self.pedestrian_scan_count = 0
        self.lower_scan_thresh = 3
        self.upper_scan_thresh = 20
        self.crosswalk_turn_buffer = 0

    def image_callback(self, data):
        movement = Twist()

        current_camera_image = np.empty(ROAD_IMAGE_SHAPE)
        try:
            current_camera_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        road_image = process_road(current_camera_image)
        self.current_road_image = road_image

        if self.crosswalk_turn_buffer <= 0:
            crosswalk_score = process_crosswalk(current_camera_image)

            if crosswalk_score >= CROSSWALK_STOP_THRESH:
                movement.linear.x = 0
                movement.angular.z = 0

                self.vel_pub.publish(movement)

                if not self.pedestrian_scan:
                    self.pedestrian_scan = True
                    self.crosswalk_turn_buffer = CROSSWALK_TURN_BUFFER
                    self.first_crosswalk_image = current_camera_image

        else:
            crosswalk_score = 0

        if self.pedestrian_scan:
            pedestrian_score = process_pedestrian(self.first_crosswalk_image, current_camera_image)
            if self.lower_scan_thresh < pedestrian_score < self.upper_scan_thresh:
                self.pedestrian_scan = False
                self.lower_scan_thresh = 10
                self.pedestrian_scan_count = 0

                # TODO: implement rolling average check
            else:
                if self.pedestrian_scan_count >= SECOND_PEDESTRIAN_COUNT_THRESH:
                    self.lower_scan_thresh = SECOND_LOWER_PEDESTRIAN_THRESH
                    self.upper_scan_thresh = SECOND_UPPER_PEDESTRIAN_THRESH
                    self.first_crosswalk_image = current_camera_image
                self.pedestrian_scan_count += 1
                return

        movement_prediction = self.av_model.predict(reshape(road_image, (1, 108, 192, 1)), verbose=0)[0]
        prediction_state = np.argmax(movement_prediction)

        movement.linear.x = LINEAR_SPEED

        if prediction_state == 0:
            movement.angular.z = 0

        elif prediction_state == 1:
            movement.angular.z = ANGULAR_SPEED
            self.crosswalk_turn_buffer -= 1

        elif prediction_state == 2:
            movement.angular.z = -1 * ANGULAR_SPEED
            self.crosswalk_turn_buffer -= 1

        else:
            movement.angular.z = 0

        try:
            self.vel_pub.publish(movement)
        except Exception as e:
            print(e)

        return
