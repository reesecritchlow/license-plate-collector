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
VIDEO_SECS = 1
FPS = 50
SHAPE = (108, 192)

LINEAR_SPEED = 1.743392200500000766e-01
ANGULAR_SPEED = 9.000000000000000222e-01

class turn_injector:
    """
    Class used to collect image data of the course and label each image with its respective command velocity.
    """
    def __init__ (self, data_name):
        self.bridge = CvBridge()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(f'{VID_LOCATION}{data_name}.mp4', fourcc, FPS, (SHAPE[1], SHAPE[0]), 0)

        self.vel_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.twist_callback)
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.image_callback)
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)

        self.vel_data = np.empty((0,2))
        self.twist = Twist()
        self.released = False
        self.captured = False

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            processed_image = process_image(cv_image)
        except CvBridgeError as e:  
            print(e)

        movement = Twist()

        if not self.captured:
            # movement.linear.x = -1 * self.twist.linear.x * 0.5

            # if self.twist.angular.z != 0:
            #     movement.angular.z = -1 * self.twist.angular.z * 0.5
            # else:
            #     movement.angular.z = 0

            # self.vel_pub.publish(movement)
            self.keyframe = processed_image
            self.captured = True

        else:
            if not self.released:
                for _ in range(10):
                    movement.angular.z = -1 
                    movement.linear.x = 1
                    self.vel_pub.publish(movement)               

                for _ in range(50):
                    self.video_writer.write(self.keyframe)
                    self.vel_data = np.append(self.vel_data, [[-1 * ANGULAR_SPEED, LINEAR_SPEED]], axis=0)
                    
                self.released = True
                print("done")
                self.video_writer.release()
                print("released")
                print(f'{self.twist.angular.z}, {self.twist.linear.x}')

                rospy.signal_shutdown('swag')
            
        cv2.imshow('pov', processed_image)
        cv2.waitKey(3)
    
    def twist_callback(self, data):
        self.twist = data


def main(args):
    data_name = input("Data name (NO .mp4): ") 
    rospy.init_node('inj', anonymous=True)
    dc = turn_injector(data_name)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    np.savetxt(f'/home/rcritchlow/ENPH353_Team16_Data/{data_name}.csv', dc.vel_data, delimiter=',')
    cv2.destroyAllWindows()

main(sys.argv)