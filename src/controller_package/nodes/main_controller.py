#!/usr/bin/env python

import rospy
import sys
import timer_controller
import direct_controller
import cv2

from imitation_controller import ImitationController


def main(args):
    tc = timer_controller.timer_controller()
    dc = direct_controller.direct_controller()
    rospy.init_node('main_controller', anonymous=True)

    tc.start()
    dc.spin(-90)
    dc.drive(0.4, 0.3)
    dc.spin(90)
    # dc.drive(1, 0.3)
    ic = ImitationController(tc)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
