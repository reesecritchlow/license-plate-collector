#!/usr/bin/env python

import rospy
import sys
import cv2

from timer_controller import TimerController
from direct_controller import DirectController
from outside_controller import OutsideController


def main(args):
    tc = TimerController()
    dc = DirectController()
    rospy.init_node('main_controller', anonymous=True)

    tc.start()
    dc.spin(-90)
    dc.drive(0.4, 0.3)
    dc.spin(90)
    # dc.drive(1, 0.3)
    oc = OutsideController(tc)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
