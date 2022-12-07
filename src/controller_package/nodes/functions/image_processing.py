#!/usr/bin/env python

import cv2
import numpy as np
import uuid


def process_road(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([195, 35, 255])
    white_space = cv2.inRange(hsv, lower_white, upper_white)
    new_dims = (192, 108)
    resized = cv2.resize(white_space, new_dims, interpolation=cv2.INTER_AREA)
    return resized


def process_crosswalk(image):
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([110, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    line_count = np.sum(mask[-1] != 255)

    return line_count


def process_pedestrian(initial, current):
    gray_initial = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    difference = cv2.subtract(gray_initial, gray_current)
    _, thresh = cv2.threshold(difference, 70, 255, cv2.THRESH_BINARY)

    center = thresh[:, range(len(thresh[0]) // 2 - 5, len(thresh[0]) // 2 + 5)]
    center_count = np.sum(center != 0)

    cv2.imshow('image', thresh)
    cv2.waitKey(3)

    print(center_count)

    return center_count
