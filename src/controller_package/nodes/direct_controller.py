#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import time
import numpy as np

BASE_ANGULAR_VELOCITY = np.pi / 4 # rad/sec

class direct_controller:
    def __init__(self):
        self.movement_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    
    def spin(self, angle, scaling=1):
        movement = Twist()
        angle_radians = angle * np.pi / 180
        delay = angle_radians / (BASE_ANGULAR_VELOCITY * scaling)
        direction = 1 if angle > 0 else -1

        movement.angular.z = 1 * scaling * direction
        self.movement_pub.publish(movement)
        time.sleep(np.abs(delay))
        movement.angular.z = 0
        self.movement_pub.publish(movement)
        return


    def drive(self, distance, speed=0.5):
        movement = Twist()
        delay = distance / speed
        direction = 1 if distance >= 0 else -1

        movement.linear.x = speed * direction
        self.movement_pub.publish(movement)
        time.sleep(np.abs(delay))
        movement.linear.x = 0
        self.movement_pub.publish(movement)
        
        return
   