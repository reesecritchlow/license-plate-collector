#!/usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class data_collector:
    def __init__ (self):
        self.bridge = CvBridge()
        self.vel_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.twist_callback)
        self.twist = Twist()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:  
            print(e)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
        print(self.twist.linear.x)

        # try:
        #     self.cmd_pub.publish(move)
        # except CvBridgeError as e:
        #     print(e)
    
    def twist_callback(self, data):
        self.twist = data


def main(args):
    dc = data_collector()
    rospy.init_node('data collection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

main(sys.argv)