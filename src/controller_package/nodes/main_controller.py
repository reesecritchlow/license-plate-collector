#!/usr/bin/env python

import roslib
import rospy
import sys
import cv2
from license_identification import license_detector


def main(args):
    
    rospy.init_node('main_controller', anonymous=False)
    char_collect = input("Character collection? (Y/N) ")
    if char_collect: 
        chars_in_view = input("Plate characters (AB12): ")
        save_number = input("Save number (A1, B1, 11, 21)")
        ld = license_detector(input("plate_number:"), collect_data=True)
    else:
        ld = license_detector(input("plate_number:"), collect_data=False)

    

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows

if __name__ == '__main__':
    main(sys.argv)
