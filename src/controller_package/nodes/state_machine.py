#!/usr/bin/env python

"""
state_machine.py:

Main state machine for robot control.
"""

import string

from collections import deque

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from tensorflow.keras import models
from tensorflow import reshape
import threading
import uuid
import os
from dotenv import load_dotenv
import utils.image_processing as ip

load_dotenv()
FILE_PATH = os.getenv('COMP_DIRECTORY')

# Neural network driving speeds for the outside circuit
LINEAR_SPEED = 1.743392200500000766e-01
ANGULAR_SPEED = 9.000000000000000222e-01

# Neural network driving speeds for the inside circuit
INSIDE_ANGULAR_SPEED = 1.265767756416901202e+00
INSIDE_LINEAR_SPEED = 2.952450000000000907e-01

CROSSWALK_STOP_THRESH = 400
CROSSWALK_TURN_BUFFER = 20  # number of turn actions to pass before looking for another crosswalk

SECOND_PEDESTRIAN_COUNT_THRESH = 80  # number of pedestrian samples before resampling
SECOND_LOWER_PEDESTRIAN_THRESH = 1
SECOND_UPPER_PEDESTRIAN_THRESH = 10000

PEDESTRIAN_QUEUE_SIZE = 5
QUEUE_DEVIANCE = 2 * 8

COL_CROP_RATIO = 5 / 8
ROW_RATIO = 3 / 8

MIN_PLATE_AREA = 8_000
MAX_PLATE_AREA = 30_000

PLATE_THREAD_WINDOW = 60

ROAD_IMAGE_SHAPE = (192, 108)

class StateMachine:
    """_summary_
    StateMachine: 

    Creates a control system for the robot in the ENPH 353 competition.
    """
    def __init__(self, timer):
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.timer = timer
        self.av_model = models.load_model(f'/home/{FILE_PATH}/src/controller_package/models/rm5_modified_10.h5')
        self.license_model = models.load_model(f'/home/{FILE_PATH}/src/controller_package/models/license_model_v2.h5')
        self.parking_model = models.load_model(f'/home/{FILE_PATH}/src/controller_package/models/parking_model.h5')
        self.inside_model = models.load_model(f'/home/{FILE_PATH}/src/controller_package/nodes/inner_model_3.h5')

        self.drive_model = self.av_model
        self.inside = False

        self.current_road_image = []
        self.image_stream = []
        self.first_crosswalk_image = []

        self.pedestrian_scan = False
        self.pedestrian_scan_count = 0
        self.lower_scan_thresh = 3
        self.upper_scan_thresh = 20
        self.crosswalk_turn_buffer = 0
        self.pedestrian_queue = deque(maxlen=PEDESTRIAN_QUEUE_SIZE)

        self.collect_data = False
        self.chars_in_view = "AAAA"
        self.save_number = "0"
        self.max_frames = 300
        self.in_light = False

        self.frame_counter = 0
        self.max_area = 0
        self.plate_save = False

        self.predicted = False
        self.last_plate = None
        self.last_parking = None

        self.plate_window_count = 0
        self.plate_window_open = False
        self.allow_count = False
        self.plate_thread = threading.Thread()

        self.plate_positions = deque(['2', '3', '4', '5', '6', '1'])

        self.reverse_dic = ip.reverse_dictionary()

    def data_image_callback(self, data):
        self.frame_counter += 1
        print(self.frame_counter)
        if self.frame_counter <= self.max_frames:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

            processed_img = ip.process_image(cv_image)
            cv_image = ip.crop_camera(cv_image)

            contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            matrix = None
            front_approx = None

            if len(contours) > 0:
                front_approx, _ = ip.get_front_approx(cv_image, contours)

            front_perspective = ip.get_front_perspective(cv_image, front_approx)
            license_plate, chars, combined_chars, parking_spot = ip.get_plate(front_perspective)

            if not self.in_light:
                for i, label in enumerate(self.chars_in_view):
                    # cv2.imwrite(f"/home/fizzer/data/images/test/{label}{self.save_number}.png", chars[i])
                    cv2.imwrite(f"/home/fizzer/data/images/parking/{label}{self.save_number}.png", parking_spot)

            else:
                for i, label in enumerate(self.chars_in_view):
                    cv2.imwrite(f"/home/fizzer/data/images/test/light/{label}{self.save_number}.png", chars[i])

            self.save_number += 1
        else:
            print('DONE COLLECTING DATA')
            rospy.signal_shutdown('Finished collecting data.')

    def predict(self, thread_name):
        predict_0 = self.license_model.predict(np.expand_dims(gray_scale(self.last_plate[0]), axis=0))[0]
        predict_1 = self.license_model.predict(np.expand_dims(gray_scale(self.last_plate[1]), axis=0))[0]
        predict_2 = self.license_model.predict(np.expand_dims(gray_scale(self.last_plate[2]), axis=0))[0]
        predict_3 = self.license_model.predict(np.expand_dims(gray_scale(self.last_plate[3]), axis=0))[0]

        parking_predict = self.parking_model.predict(np.expand_dims(self.last_parking, axis=0))[0]

        i0, = np.where(np.isclose(predict_0, 1.))
        i1, = np.where(np.isclose(predict_1, 1.))
        i2, = np.where(np.isclose(predict_2, 1.))
        i3, = np.where(np.isclose(predict_3, 1.))

        ip = np.where(np.isclose(parking_predict, 1.))

        l1 = self.reverse_dic[i0[0]]
        l2 = self.reverse_dic[i1[0]]
        n3 = self.reverse_dic[i2[0]]
        n4 = self.reverse_dic[i3[0]]

        if l1.isnumeric():
            l1 = ip.numeric_mismatch(l1)

        if l2.isnumeric():
            l2 = ip.numeric_mismatch(l2)

        if n3.isalpha():
            n3 = ip.alpha_mismatch(n3)

        if n4.isalpha():
            n4 = ip.alpha_mismatch(n4)

        print(
            f"PREDICT {l1}{l2}{n3}{n4}")
        print(f"PARKING PREDICT: {ip[0] + 1}")

        self.timer.publish_plate(f'{(ip[0] + 1)[0]}', f'{l1}{l2}{n3}{n4}')
        print(f'{thread_name} finished executing.')

        if str((ip[0] + 1)[0]) == '1':
            self.timer.terminate()

        return

    def image_callback(self, data):

        global cv_image
        current_camera_image = np.empty(ROAD_IMAGE_SHAPE)
        try:
            current_camera_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = current_camera_image
        except CvBridgeError as e:
            print(e)

        movement = Twist()

        road_image = ip.process_road(current_camera_image)
        self.current_road_image = road_image

        if self.pedestrian_scan:
            pedestrian_score = ip.process_pedestrian(self.first_crosswalk_image, current_camera_image)

            if len(self.pedestrian_queue) == PEDESTRIAN_QUEUE_SIZE:
                if sum(self.pedestrian_queue) / len(
                        self.pedestrian_queue) - QUEUE_DEVIANCE >= pedestrian_score or pedestrian_score >= sum(
                    self.pedestrian_queue) / len(self.pedestrian_queue) + QUEUE_DEVIANCE:
                    self.pedestrian_scan = False
                    self.pedestrian_scan_count = 0
                    self.pedestrian_queue.clear()
                else:
                    self.pedestrian_queue.append(pedestrian_score)
                    if self.pedestrian_scan_count >= SECOND_PEDESTRIAN_COUNT_THRESH:
                        self.first_crosswalk_image = current_camera_image
                        self.pedestrian_scan_count = 0
                        self.pedestrian_queue.clear()
                    self.pedestrian_scan_count += 1
                    return
            else:
                self.pedestrian_queue.append(pedestrian_score)
                return

        if self.crosswalk_turn_buffer <= 0:
            crosswalk_score = ip.process_crosswalk(current_camera_image)

            if crosswalk_score >= CROSSWALK_STOP_THRESH:
                movement.linear.x = 0
                movement.angular.z = 0

                self.vel_pub.publish(movement)

                if not self.pedestrian_scan:
                    self.pedestrian_scan = True
                    self.crosswalk_turn_buffer = CROSSWALK_TURN_BUFFER
                    self.first_crosswalk_image = current_camera_image
                    return

        movement_prediction = self.drive_model.predict(reshape(road_image, (1, 108, 192, 1)), verbose=0)[0]
        prediction_state = np.argmax(movement_prediction)

        if not self.inside:
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
        else:
            if prediction_state == 0:
                movement.linear.x = 0
                movement.angular.z = 0
            elif prediction_state == 1:
                movement.linear.x = 0
                movement.angular.z = INSIDE_ANGULAR_SPEED
            elif prediction_state == 2:
                movement.linear.x = 0
                movement.angular.z = -1 * INSIDE_ANGULAR_SPEED
            elif prediction_state == 3:
                movement.linear.x = INSIDE_LINEAR_SPEED
                movement.angular.z = 0
            elif prediction_state == 4:
                movement.linear.x = INSIDE_LINEAR_SPEED
                movement.angular.z = INSIDE_ANGULAR_SPEED
            elif prediction_state == 5:
                movement.linear.x = INSIDE_LINEAR_SPEED
                movement.angular.z = -1 * INSIDE_ANGULAR_SPEED
            else:
                movement.linear.x = INSIDE_LINEAR_SPEED
                movement.angular.z = 0

        try:
            self.vel_pub.publish(movement)
        except Exception as e:
            print(e)

        processed_img = ip.process_image(cv_image)
        cv_image = ip.crop_camera(cv_image)
        contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if we see the front section on a car
        if len(contours) > 0:
            approx, max_contour = ip.get_front_approx(cv_image, contours)

            if len(approx) == 4 and cv2.contourArea(max_contour) > self.max_area:
                corner = max([(sum(pt[0]), i) for i, pt in enumerate(max_contour)])
                corner_coords = np.array([max_contour[corner[1]][0, 1],
                                          max_contour[corner[1]][0, 0]])

                self.max_area = cv2.contourArea(max_contour)

                # end of front perspective
                front_transformed = ip.get_front_perspective(cv_image, approx)

                # start of get license plate
                license_plate, chars, combined_chars, parking_spot = ip.get_plate(front_transformed)

                plate_post = ip.contour_format(license_plate)

                number_cnt, _ = cv2.findContours(plate_post, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                total_num_area = np.sum([cv2.contourArea(cnt) for cnt in number_cnt])

                if (len(number_cnt) > 0
                        and MIN_PLATE_AREA < total_num_area < MAX_PLATE_AREA
                        and corner_coords[0] != cv_image.shape[0] - 1
                        and corner_coords[1] != cv_image.shape[1] - 1
                        and corner_coords[0] != 0
                        and corner_coords[1] != 0):

                    parking_shape = parking_spot.shape

                    self.last_parking = ip.contour_format(parking_spot, blur_factor=20, threshold=80)[:,
                                        int(parking_shape[1] / 2):parking_shape[1]]
                    self.last_plate = chars
                    self.predicted = False
                    self.plate_save = True

                else:
                    if not self.predicted:
                        self.predicted = True

                        if self.plate_window_count == 0:
                            self.allow_count = True

                        if self.plate_window_count < PLATE_THREAD_WINDOW:
                            print('thread added')
                            self.plate_thread = threading.Thread(target=self.predict, args=(uuid.uuid4(),))
            else:
                self.max_area = 0

            if self.allow_count:
                self.plate_window_count += 1

            if self.plate_window_count >= PLATE_THREAD_WINDOW:
                self.plate_thread.start()
                self.plate_window_count = 0
                self.allow_count = False

        return
