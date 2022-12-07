#!/usr/bin/env python3
from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospy
from sensor_msgs.msg import Image
import sys

import os
from dotenv import load_dotenv

load_dotenv()

FILE_PATH = os.getenv('COMP_DIRECTORY')

sys.path.insert(0, f'/home/{FILE_PATH}/src/controller_package/nodes')
from image_processing import process_pedestrian

class frame_collector:
    def __init__(self, filename):
        self.counter = 0
        self.filename = filename
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)


    def image_callback(self, data):
        if self.counter == 0:
            self.original_image = self.bridge.imgmsg_to_cv2(data)
        processed = process_pedestrian(self.original_image, self.bridge.imgmsg_to_cv2(data))
        cv2.imshow('crosswalk', processed)
        cv2.waitKey(3)
        # cv2.imwrite(f'{self.filename}.jpg', processed)
        self.counter += 1
        return

myfile = input('enter a filename: ')
rospy.init_node('framer', anonymous=True)
fc = frame_collector(myfile)

rospy.spin()