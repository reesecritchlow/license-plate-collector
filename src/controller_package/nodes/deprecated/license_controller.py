#!/usr/bin/env python

import rospy
import sys
import cv2
from license_identification import license_detector


from timer_controller import TimerController
from direct_controller import DirectController
from outside_controller import OutsideController


def main(args):
    rospy.init_node('main_controller', anonymous=True)
    ld = license_detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
