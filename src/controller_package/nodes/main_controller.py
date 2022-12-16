#!/usr/bin/env python

import rospy
import sys
import cv2
from license_identification import license_detector


from utils.timer_controller import TimerController
from utils.direct_controller import DirectController
from state_machine import StateMachine


def main(args):
    rospy.init_node('main_controller', anonymous=True)
    timer = TimerController()
    dc = DirectController()
    dc.drive(0.01, 0.01)
    tc.start()
    dc.drive(0.4, 0.3)
    dc.spin(90)
    sm = StateMachine(timer)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
