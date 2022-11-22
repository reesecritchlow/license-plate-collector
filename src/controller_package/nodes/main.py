#!/usr/bin/env python

import roslib
import rospy
import sys
import timer_controller
import cv2

TEAM_ID = 'mode_push_16'

def main(args):
    tc = timer_controller.timer_controller(TEAM_ID)
    rospy.init_node('main_controller', anonymous=True)
    tc.start()
    tc.terminate()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows

if __name__ == '__main__':
    main(sys.argv)