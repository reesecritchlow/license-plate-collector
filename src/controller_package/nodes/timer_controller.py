#!/usr/bin/env python

import roslib
import rospy
from std_msgs.msg import String
import sys

import os
from dotenv import load_dotenv

STARTING_LOCATION = 0
STARTING_PLATE = 'TR88'
ENDING_LOCATION = -1
ENDING_PLATE = 'RT88'

class timer_controller:
    """
    Class for interfacing with the timer and scoreboard interface for the enph353 competition.
    """

    def __init__(self, teamID):
        load_dotenv()
        self.timer_pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.teamID = teamID
        self.team_password = os.getenv('TEAM_PASSWORD')
        print('hello')

    def generate_message_string(self, location, plate):
        """
        Generates a message string.

        Args:
            location (string): plate location
            plate (string): plate number/letters

        Returns:
            String: formatted string for interfacing with the controller
        """
        return String(f'{self.teamID},{self.team_password},{location},{plate}')

    def publish_plate(self, location, plate):
        """
        Publishes a plate to the 353 scoreboard.

        Args:
            location (number): plate location
            plate (string): plate numbers/letters
        """
        self.timer_pub.publish(self.generate_message_string(location, plate))
    
    def start(self):
        """
        Initializes the timer for enph353 competition.
        """
        self.publish_plate(STARTING_LOCATION, STARTING_PLATE)

    def terminate(self):
        """
        Stops the timer and closes out the enph353 competition.
        """
        self.publish_plate(ENDING_LOCATION, ENDING_PLATE)
