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

from image_processing import process_image

LINEAR_SPEED = 1.743392200500000766e-01
ANGULAR_SPEED = 9.000000000000000222e-01

class ImitationController:
    def __init__(self, timer):
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.timer = timer
        self.av_model = models.load_model('/home/rcritchlow/ros_ws/src/controller_package/nodes/reese_model_3.h5')

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        processed_image = process_image(cv_image)

        cv2.imshow('stream', processed_image)
        cv2.waitKey(3)

        prediction = self.av_model.predict(reshape(processed_image, (1, 108, 192, 1)), verbose=0)[0]
        i = np.argmax(prediction)

        movement = Twist()
        movement.linear.x = LINEAR_SPEED

        print(prediction)
        print(i)

        if i == 0:
            movement.angular.z = 0
        elif i == 1:
            movement.angular.z = ANGULAR_SPEED
        elif i == 2:
            movement.angular.z = -1 * ANGULAR_SPEED
        else:
            movement.angular.z = 0

        try:
            self.vel_pub.publish(movement)
        except CvBridgeError as e:
            print(e)
