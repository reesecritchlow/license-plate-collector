#!/usr/bin/env python

import roslib
import rospy
import sys
import cv2
from license_identification import license_detector


def main(args):
    rospy.init_node('data_controller', anonymous=False)
    char_collect = input("Character collection? (Y/N) ")
    ld = None
    if char_collect == 'Y': 
        chars_in_view = input("Plate characters (AB12): ")
        save_number = input("Save number (A0, B0, 10, 20)")
        frames = input("Number of frames: ")
        ld = license_detector(char_collect, chars_in_view, int(save_number), int(frames))
    else:
        ld = license_detector()

    

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows

if __name__ == '__main__':
    main(sys.argv)
