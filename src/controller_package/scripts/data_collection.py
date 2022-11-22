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

VID_NAME = "/home/fizzer/ENPH353_Team16/src/data/video_data.mp4"
VIDEO_SECS = 60
FPS = 20
SHAPE = (720,1280)

class data_collector:
    def __init__ (self):
        self.bridge = CvBridge()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(VID_NAME, fourcc, FPS, (SHAPE[1], SHAPE[0]))
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
        except CvBridgeError as e:  
            print(e)

        if self.frame_count < FPS*VIDEO_SECS:
            if self.frame_count % 3 == 0:
                print(f"{self.frame_count}, {FPS*VIDEO_SECS}")
                self.video_writer.write(cv_image)
                self.vel_data = np.append(self.vel_data, [[self.twist.angular.z, self.twist.linear.x]], axis=0)
        else:
            if not self.released:
                self.released = True
                print("done")
                self.video_writer.release()

    
    def twist_callback(self, data):
        self.twist = data


def main(args):
    dc = data_collector()
    rospy.init_node('data collection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    np.savetxt('/home/fizzer/ENPH353_Team16/src/data/data.csv', dc.vel_data, delimiter=',')
    cv2.destroyAllWindows()

main(sys.argv)