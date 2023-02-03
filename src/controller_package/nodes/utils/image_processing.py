#!/usr/bin/env python

import cv2
import numpy as np
import string
from scipy.spatial import distance as dist

LOWER_WHITE = np.array([0, 0, 86], dtype=np.uint8)
UPPER_WHITE = np.array([127, 17, 206], dtype=np.uint8)
MIN_AREA = 8_000
MAX_AREA = 28_000
WIDTH = 600
HEIGHT = 1200
PERSPECTIVE_OUT = np.float32([[0, 0], [0, HEIGHT - 1], [WIDTH - 1, HEIGHT - 1], [WIDTH - 1, 0]])

"""
Set of helper functions for processing images for various points in the course.
"""


def process_road(image):
    """
    process_road:
    processes an image from the robot, returns an image binarized to the road limit lines,
    resized down to 192x108px.

    Args:
        image (numpy.ndarray): input image from ROS Robot

    Returns:
        numpy.ndarray: processed image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([195, 35, 255])
    white_space = cv2.inRange(hsv, lower_white, upper_white)
    new_dims = (192, 108)
    resized = cv2.resize(white_space, new_dims, interpolation=cv2.INTER_AREA)
    return resized


def process_crosswalk(image):
    """process_crosswalk:

    Args:
        image (numpy.ndarray): image from ROS camera

    Returns:
        int: number of red pixels in the bottom row of the image
    """
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([110, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    line_count = np.sum(mask[-1] != 255)

    return line_count


CROSSWALK_BAND = 10


def process_pedestrian(initial, current):
    """process_pedestrian

    Args:
        initial (numpy.ndarray): initial image of the crosswalk that the robot captured
        current (numpy.ndarray): current image of the crosswalk that the robot has captured

    Returns:
        int: count of pixels in the middle 20 rows of the image that have a white object present in them
    """
    gray_initial = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    difference = cv2.subtract(gray_initial, gray_current)
    _, thresh = cv2.threshold(difference, 70, 255, cv2.THRESH_BINARY)

    center = thresh[:, range(len(thresh[0]) // 2 - CROSSWALK_BAND, len(thresh[0]) // 2 + CROSSWALK_BAND)]
    center_count = np.sum(center != 0)

    return center_count


def reverse_dictionary():
    reverse_dic = {}

    for i in range(26):
        reverse_dic[i] = string.ascii_uppercase[i]

    for i in range(10):
        reverse_dic[i + 26] = str(i)

    return reverse_dic


def corner_fix(contour):
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


def contour_format(image, blur_factor=7, threshold=10, lower=np.array([0, 0, 0]),
                   upper=np.array([144, 85, 255])):
    """Applies threshold, blur, and grayscale to OpenCV image.

    Args:
        image (np.darray): image to be formatted
        blur_factor (int, optional): blur kernel size. Defaults to 7.
        threshold (int, optional): min threshold value . Defaults to 10.
        lower (np.array, optional): lower mask bound. Defaults to np.array([0, 0, 0]).
        upper (np.array, optional): upper mask bound. Defaults to np.array([144, 85, 255]).

    Returns:
        _type_: _description_
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    split = cv2.split(result)[2]
    blur = cv2.blur(split, (blur_factor, blur_factor))
    thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    return thresh


def process_image(image):
    """Processes image for license identification

    Args:
        image (np.darray): image to be processed

    Returns:
        np.darray: processed image
    """
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


def crop_camera(image):
    """crops image

    Args:
        image (np.darray): image to be cropped

    Returns:
        np.darray: cropped image
    """
    rows = image.shape[0]
    cols = image.shape[1]
    image = image[int(2 / 5 * rows):int(4 / 5 * rows), 0:cols]
    return image


def get_front_approx(image, contours):
    """Creates polygon contour to approximate the edges of a contour.

    Args:
        image (np.darray): image with contour
        contours (np.darray): original contours seen on image

    Returns:
        tuple: the contour of the approximation, the largest contour in the original contours
    """
    c = max(contours, key=cv2.contourArea)

    if MAX_AREA > cv2.contourArea(c) > MIN_AREA:
        cv2.drawContours(image, [c], 0, (0, 0, 255), 3)

        # find, and draw approximate polygon for contour c
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * 0.05, True)[0:4]

        return approx, c

    return [], []


def get_front_perspective(image, approx_contours):
    """Transforms a polygon to a rectangle, perspective shifts the image inside the polygon such that it is orthogonal to the screen. 

    Args:
        image (np.darray): image with license plate contour
        approx_contours (np.darray): approximate polygon contour

    Returns:
        np.darray: Perspective shifted image
    """
    perspective_in = corner_fix(approx_contours)
    cv2.drawContours(image, [approx_contours], 0, (0, 255, 0), 3)

    # matrix transformation for perpective shift of license plate
    matrix = cv2.getPerspectiveTransform(perspective_in, PERSPECTIVE_OUT)
    imgOutput = cv2.warpPerspective(image, matrix, (WIDTH, HEIGHT), cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return imgOutput


def get_plate(image):
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


def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def numeric_mismatch(character):
    if int(character) == 5:
        character = 'S'
    elif int(character) == 2:
        character = 'Z'
    elif int(character) == 1:
        character = 'I'
    elif int(character) == 8:
        character = 'B'

    return character


def alpha_mismatch(character):
    if character == 'S':
        character = '5'
    elif character == 'Z':
        character = '2'
    elif character == 'I':
        character = '1'
    elif character == 'B':
        character = '8'
    elif character == 'R':
        character = '8'

    return character
