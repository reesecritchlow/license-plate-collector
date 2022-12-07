# This is a script taken from online to find HSV thresholds
# https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv

import cv2
import numpy as np

def nothing(x):
    pass

# Load image
image = cv2.imread('images/plate6.png')

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
cv2.createTrackbar('blur', 'image', 1, 9, nothing)
cv2.createTrackbar('thresh_min', 'image', 0, 255, nothing)
cv2.createTrackbar('thresh_max', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMin', 'image', 0)
cv2.setTrackbarPos('SMin', 'image', 0)
cv2.setTrackbarPos('VMin', 'image', 0)
cv2.setTrackbarPos('HMax', 'image', 144)
cv2.setTrackbarPos('SMax', 'image', 85)
cv2.setTrackbarPos('VMax', 'image', 255)
cv2.setTrackbarPos('blur', 'image', 9)
cv2.setTrackbarPos('thresh_min', 'image', 58)
cv2.setTrackbarPos('thresh_max', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')
    blurnum = cv2.getTrackbarPos('blur', 'image')
    thresh_min = cv2.getTrackbarPos('thresh_min', 'image')
    thresh_max = cv2.getTrackbarPos('thresh_max', 'image')
    

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d), thresh=%d, %d" % (hMin , sMin , vMin, hMax, sMax , vMax, thresh_min, thresh_max))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax
    
    # Display result image
    split = cv2.split(result)[2]
    blur = cv2.blur(split,(blurnum,blurnum))
    thresh = cv2.threshold(blur, thresh_min, thresh_max, cv2.THRESH_BINARY_INV)[1]

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key = cv2.contourArea)
        # print(contours[0])
        # cnt = contours[4]
        cv2.drawContours(result, contours, -1, (0,0,255), 3)


    cv2.imshow('split', split)

    cv2.imshow('thresh', thresh)
    cv2.imshow('image', result)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()