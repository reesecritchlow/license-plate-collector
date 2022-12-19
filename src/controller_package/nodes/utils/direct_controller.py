#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import time
import numpy as np

BASE_ANGULAR_VELOCITY = np.pi / 4  # rad/sec


class DirectController:
    """
    DirectController:

    Class responsible for executing 'hardcoded' for the robot.
    """
    def __init__(self):
        self.movement_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    def spin(self, angle, scaling=1):
        """spin

        rotates the robot by an angle

        Args:
            angle (number): angle to rotate the robot by (degrees)
            scaling (int, optional): How fast to scale the rotation speed by. Defaults to 1.
        """
        movement = Twist()
        angle_radians = angle * np.pi / 180
        delay = angle_radians / (BASE_ANGULAR_VELOCITY * scaling)
        direction = 1 if angle > 0 else -1

        movement.angular.z = 1 * scaling * direction
        self.movement_pub.publish(movement)
        time.sleep(np.abs(delay))
        movement.angular.z = 0
        self.movement_pub.publish(movement)
        time.sleep(0.5)
        return

    def drive(self, distance, speed=0.5):
        """drive:

        Args:
            distance (int): Distance in m to drive the robot by 
            speed (float, optional): scaling speed for linear movement. Defaults to 0.5.
        """
        movement = Twist()
        delay = distance / speed
        direction = 1 if distance >= 0 else -1

        movement.linear.x = speed * direction
        self.movement_pub.publish(movement)
        time.sleep(np.abs(delay))
        movement.linear.x = 0
        self.movement_pub.publish(movement)
        time.sleep(0.5)
        return
