#!/usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import sys

sys.path.insert(0, '/home/rcritchlow/ros_ws/src/controller_package/nodes')

from image_processing import process_image

VID_LOCATION = "/home/rcritchlow/ENPH353_Team16_Data/"
VIDEO_SECS = 120
FPS = 5
SHAPE = (108, 192)

class data_collector:
    """
    Class used to collect image data of the course and label each image with its respective command velocity.
    """
    def __init__ (self, data_name):
        self.bridge = CvBridge()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(f'{VID_LOCATION}{data_name}.mp4', fourcc, FPS, (SHAPE[1], SHAPE[0]), 0)
        self.frame_count = 0

        self.vel_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.twist_callback)
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)

        self.vel_data = np.empty((0,2))
        self.twist = Twist()
        self.released = False

    def image_callback(self, data):
        self.frame_count += 1
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            processed_image = process_image(cv_image)
        except CvBridgeError as e:  
            print(e)

        if self.frame_count < FPS*VIDEO_SECS:
            print(f"{self.frame_count}, {FPS*VIDEO_SECS}")
            print(processed_image.shape)
            self.video_writer.write(processed_image)
            self.vel_data = np.append(self.vel_data, [[self.twist.angular.z, self.twist.linear.x]], axis=0)
        else:
            if not self.released:
                self.released = True
                print("done")
                self.video_writer.release()
                print("released")
            
        cv2.imshow('pov', processed_image)
        cv2.waitKey(3)
    
    def twist_callback(self, data):
        self.twist = data


def main(args):
    data_name = input("Data name (NO .mp4): ") 
    rospy.init_node('data_collection', anonymous=True)
    dc = data_collector(data_name)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    np.savetxt(f'/home/rcritchlow/ENPH353_Team16_Data/{data_name}.csv', dc.vel_data, delimiter=',')
    cv2.destroyAllWindows()

main(sys.argv)