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
    def __init__(self, teamID):
        load_dotenv()
        self.timer_pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.teamID = teamID
        self.team_password = os.getenv('TEAM_PASSWORD')
        print('hello')

    def generate_message_string(self, location, plate):
        return String(f'{self.teamID},{self.team_password},{location},{plate}')
    
    def start(self):
        try:
            self.timer_pub.publish(self.generate_message_string(STARTING_LOCATION, STARTING_PLATE))
        except Exception as e: 
            print(e)

    def terminate(self):
        self.timer_pub.publish(self.generate_message_string(ENDING_LOCATION, ENDING_PLATE))

    def publish_plate(self, location, plate):
        self.timer_pub.publish(self.generate_message_string(location, plate))
