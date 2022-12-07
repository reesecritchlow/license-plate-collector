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
    tc = TimerController()
    dc = DirectController()
    # ld = license_detector()
    # dc.spin(-90)
    dc.drive(0.01, 0.01)
    tc.start()
    dc.drive(0.4, 0.3)
    dc.spin(90)
    oc = OutsideController(tc)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
