#!/usr/bin/env python
import string

from collections import deque

from scipy.spatial import distance as dist
import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from tensorflow.keras import models
from tensorflow import reshape
import string

import threading

import uuid

from functions.image_processing import process_road, process_crosswalk, process_pedestrian

LINEAR_SPEED = 1.743392200500000766e-01 
ANGULAR_SPEED = 9.000000000000000222e-01 

INSIDE_ANGULAR_SPEED = 1.265767756416901202e+00	
INSIDE_LINEAR_SPEED = 2.952450000000000907e-01

CROSSWALK_STOP_THRESH = 400
CROSSWALK_TURN_BUFFER = 20  # number of turn actions to pass before looking for another crosswalk

SECOND_PEDESTRIAN_COUNT_THRESH = 80  # number of pedestrian samples before resampling
SECOND_LOWER_PEDESTRIAN_THRESH = 1
SECOND_UPPER_PEDESTRIAN_THRESH = 10000

PEDESTRIAN_QUEUE_SIZE = 5
QUEUE_DEVIANCE = 2 * 8

LOWER_WHITE = np.array([0,0,86], dtype=np.uint8)
UPPER_WHITE = np.array([127,17,206], dtype=np.uint8)
# LOWER_WHITE = np.array([96,0,70], dtype=np.uint8)
# UPPER_WHITE = np.array([125,82,200], dtype=np.uint8)

COL_CROP_RATIO = 5/8
ROW_RATIO = 3/8
MIN_AREA = 8_000
MAX_AREA = 28_000
MIN_PLATE_AREA = 8_000
MAX_PLATE_AREA = 30_000

PLATE_THREAD_WINDOW = 60

WIDTH = 600
HEIGHT = 1200
PERSPECTIVE_OUT = np.float32([[0,0], [0,HEIGHT-1], [WIDTH-1,HEIGHT-1], [WIDTH-1,0]])

ROAD_IMAGE_SHAPE = (192, 108)

import os
from dotenv import load_dotenv

load_dotenv()

FILE_PATH = os.getenv('COMP_DIRECTORY')

class OutsideController:
    def __init__(self, timer):
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.timer = timer
        self.av_model = models.load_model(f'/home/{FILE_PATH}/src/controller_package/nodes/rm5_modified_10.h5')
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

        self.reverse_dic = self.reverse_dictionary()


    def reverse_dictionary(self):
        # alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        reverse_dic = {}

        for i in range(26):
            reverse_dic[i] = string.ascii_uppercase[i]

        for i in range(10):
            reverse_dic[i + 26] = str(i)

        return reverse_dic

    def corner_fix(self, contour, tolerance=10):
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


    def contour_format(self, image, blur_factor=7, threshold=10, lower=np.array([0, 0, 0]),
                       upper=np.array([144, 85, 255])):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        split = cv2.split(result)[2]
        blur = cv2.blur(split, (blur_factor, blur_factor))
        thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)[1]

        return thresh

    def process_image(self, image):
        rows = image.shape[0]
        cols = image.shape[1]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
        res = cv2.bitwise_and(image, image, mask=mask)

        split = cv2.split(res)[2]
        blur = cv2.blur(split, (5, 5))
        thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]

        crop = thresh[int(2 / 5 * rows):int(4 / 5 * rows), 0:cols]

        return crop

    def crop_camera(self, image):
        rows = image.shape[0]
        cols = image.shape[1]
        image = image[int(2 / 5 * rows):int(4 / 5 * rows), 0:cols]
        return image

    def get_front_approx(self, image, contours):
        c = max(contours, key=cv2.contourArea)

        if (cv2.contourArea(c) < MAX_AREA
                and cv2.contourArea(c) > MIN_AREA):
            cv2.drawContours(image, [c], 0, (0, 0, 255), 3)

            # find, and draw approximate polygon for contour c
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, peri * 0.05, True)[0:4]

            return approx, c

        return [], []

    def get_front_perspective(self, image, approx_contours):
        perspective_in = self.corner_fix(approx_contours)
        cv2.drawContours(image, [approx_contours], 0, (0, 255, 0), 3)

        # matrix transformation for perpective shift of license plate
        matrix = cv2.getPerspectiveTransform(perspective_in, PERSPECTIVE_OUT)
        imgOutput = cv2.warpPerspective(image, matrix, (WIDTH, HEIGHT), cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return imgOutput

    def get_plate(self, image):
        """gets license plate from perspective shifted front of car.

        Args:
            image (np.array): perspective shifted image of the front of a car

        Returns:
            license_plate: license plate image
            chars: tuple of character images
            combined_chars: combined image of characters
            parking_spot: parking number
        """
        license_plate = image[int(HEIGHT * 0.7):int(HEIGHT * 0.87), 15:WIDTH - 15]
        start = 35
        chars = (license_plate[20:184, start:start + 100],
                 license_plate[20:184, start + 100:start + 200],
                 license_plate[20:184, start + 300:start + 400],
                 license_plate[20:184, start + 400:start + 500])
        parking_spot = image[int(HEIGHT * 0.24):int(HEIGHT * 0.68), 15:WIDTH - 15]

        combined_chars = np.concatenate((chars[0], chars[1], chars[2], chars[3]), axis=1)
        return license_plate, chars, combined_chars, parking_spot

    def gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def data_image_callback(self, data):
        self.frame_counter += 1
        print(self.frame_counter)
        if self.frame_counter <= self.max_frames:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

            processed_img = self.process_image(cv_image)
            cv_image = self.crop_camera(cv_image)

            contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            matrix = None
            front_approx = None

            if len(contours) > 0:
                front_approx, _ = self.get_front_approx(cv_image, contours)

            front_perspective = self.get_front_perspective(cv_image, front_approx)
            license_plate, chars, combined_chars, parking_spot = self.get_plate(front_perspective)

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
        predict_0 = self.license_model.predict(np.expand_dims(self.gray_scale(self.last_plate[0]), axis=0))[0]
        predict_1 = self.license_model.predict(np.expand_dims(self.gray_scale(self.last_plate[1]), axis=0))[0]
        predict_2 = self.license_model.predict(np.expand_dims(self.gray_scale(self.last_plate[2]), axis=0))[0]
        predict_3 = self.license_model.predict(np.expand_dims(self.gray_scale(self.last_plate[3]), axis=0))[0]

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

        # s5 z2 i1 t1 b8

        if l1.isnumeric():
            if int(l1) == 5:
                l1 = 'S'
            elif int(l1) == 2:
                l1 = 'Z'
            elif int(l1) == 1:
                l1 = 'I'
            elif int(l1) == 8:
                l1 = 'B'
        
        if l2.isnumeric():
            if int(l2) == 5:
                l2 = 'S'
            elif int(l2) == 2:
                l2 = 'Z'
            elif int(l2) == 1:
                l2 = 'I'
            elif int(l2) == 8:
                l2 = 'B'

        if n3.isalpha():
            if n3 == 'S':
                n3 = '5'
            elif n3 == 'Z':
                n3 = '2'
            elif n3 == 'I':
                n3 = '1'
            elif n3 == 'B':
                n3 = '8'
            elif n3 == 'R':
                n3 = '8'

        if n4.isalpha():
            if n4 == 'S':
                n4 = '5'
            elif n4 == 'Z':
                n4 = '2'
            elif n4 == 'I':
                n4 = '1'
            elif n4 == 'B':
                n4 = '8'

        print(
            f"PREDICT {l1}{l2}{n3}{n4}")
        print(f"PARKING PREDICT: {ip[0]+1}")



        self.timer.publish_plate(f'{(ip[0]+1)[0]}', f'{l1}{l2}{n3}{n4}')
        print(f'{thread_name} finished executing.')

        if str((ip[0]+1)[0]) == '1':
            self.timer.terminate()
        
        return

    def image_callback(self, data):

        current_camera_image = np.empty(ROAD_IMAGE_SHAPE)
        try:
            current_camera_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = current_camera_image
        except CvBridgeError as e:
            print(e)

        movement = Twist()

        road_image = process_road(current_camera_image)
        self.current_road_image = road_image

        if self.pedestrian_scan:
            pedestrian_score = process_pedestrian(self.first_crosswalk_image, current_camera_image)

            if len(self.pedestrian_queue) == PEDESTRIAN_QUEUE_SIZE:
                print('queue average:', sum(self.pedestrian_queue)/len(self.pedestrian_queue))

                if sum(self.pedestrian_queue)/len(self.pedestrian_queue) - QUEUE_DEVIANCE >= pedestrian_score or pedestrian_score >= sum(self.pedestrian_queue)/len(self.pedestrian_queue) + QUEUE_DEVIANCE:
                    # print('escaped')
                    self.pedestrian_scan = False
                    self.pedestrian_scan_count = 0
                    self.pedestrian_queue.clear()
                else:
                    self.pedestrian_queue.append(pedestrian_score)
                    if self.pedestrian_scan_count >= SECOND_PEDESTRIAN_COUNT_THRESH:
                        self.first_crosswalk_image = current_camera_image
                        self.pedestrian_scan_count = 0
                        self.pedestrian_queue.clear()
                        # print('reset ped image')
                    self.pedestrian_scan_count += 1
                    return
            else:
                self.pedestrian_queue.append(pedestrian_score)
                return

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
                    return


            # if self.lower_scan_thresh < pedestrian_score < self.upper_scan_thresh:
            #     self.pedestrian_scan = False
            #     self.lower_scan_thresh = 10
            #     self.pedestrian_scan_count = 0
            #
            #     # TODO: implement rolling average check
            # else:
            #     if self.pedestrian_scan_count >= SECOND_PEDESTRIAN_COUNT_THRESH:
            #         self.lower_scan_thresh = SECOND_LOWER_PEDESTRIAN_THRESH
            #         self.upper_scan_thresh = SECOND_UPPER_PEDESTRIAN_THRESH
            #         self.first_crosswalk_image = current_camera_image
            #     self.pedestrian_scan_count += 1
            #     return

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
            print(prediction_state)
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

        processed_img = self.process_image(cv_image)
        cv_image = self.crop_camera(cv_image)
        contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        matrix = None

        # if we see the front section on a car
        if len(contours) > 0:
            approx, max_contour = self.get_front_approx(cv_image, contours)

            if len(approx) == 4 and cv2.contourArea(max_contour) > self.max_area:
                corner = max([(sum(pt[0]), i) for i, pt in enumerate(max_contour)])
                corner_coords = np.array([max_contour[corner[1]][0, 1],
                                          max_contour[corner[1]][0, 0]])

                self.max_area = cv2.contourArea(max_contour)

                front_transformed = self.get_front_perspective(cv_image, approx)
                # end of front perspective

                # start of get license plate
                license_plate, chars, combined_chars, parking_spot = self.get_plate(front_transformed)

                plate_post = self.contour_format(license_plate)

                number_cnt, _ = cv2.findContours(plate_post, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                total_num_area = np.sum([cv2.contourArea(cnt) for cnt in number_cnt])

                if (len(number_cnt) > 0
                        and MIN_PLATE_AREA < total_num_area < MAX_PLATE_AREA
                        and corner_coords[0] != cv_image.shape[0] - 1
                        and corner_coords[1] != cv_image.shape[1] - 1
                        and corner_coords[0] != 0
                        and corner_coords[1] != 0):

                    parking_shape = parking_spot.shape

                    self.last_parking = self.contour_format(parking_spot, blur_factor=20, threshold=80)[:, int(parking_shape[1]/2):parking_shape[1]]
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
                            # if not self.inside and len(self.plate_positions) == 1:
                            #     print('state transition')
                            #     self.inside = True
                            #     self.drive_model = self.inside_model
                                

                            
            else:
                self.max_area = 0

            if self.allow_count:
                self.plate_window_count += 1

            if self.plate_window_count >= PLATE_THREAD_WINDOW:
                self.plate_thread.start()
                self.plate_window_count = 0
                self.allow_count = False
        

        return
2