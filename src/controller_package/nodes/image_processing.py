#!/usr/bin/env python

import cv2
import numpy as np

def process_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([195, 35, 255])
    white_space = cv2.inRange(hsv, lower_white, upper_white)
    new_dims = (192, 108)
    resized = cv2.resize(white_space, new_dims, interpolation=cv2.INTER_AREA)
    return resized