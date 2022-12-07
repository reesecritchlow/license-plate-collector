#!/usr/bin/env python

import rospy
import sys
import timer_controller
import direct_controller
import cv2

from outside_controller import OutsideController


def main(args):
    tc = timer_controller.timer_controller()
    dc = direct_controller.direct_controller()
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
