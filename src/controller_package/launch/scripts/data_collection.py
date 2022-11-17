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

class data_collector:
    def __init__ (self):
        self.bridge = CvBridge()
        self.vel_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.twist_callback)
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.image_callback)
        self.vel_data = np.empty((0,2))
        self.twist = Twist()

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:  
            print(e)

        
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
        if len(self.vel_data) <= 300:
            self.vel_data = np.append(self.vel_data, [[self.twist.angular.z, self.twist.linear.x]], axis=0)
        else:
            print("done")
    
    def twist_callback(self, data):
        self.twist = data


def main(args):
    dc = data_collector()
    rospy.init_node('data collection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    np.savetxt('test.csv', dc.vel_data, delimiter=',')
    cv2.destroyAllWindows()

main(sys.argv)