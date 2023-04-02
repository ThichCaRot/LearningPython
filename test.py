import cv2
import numpy as np

cap = cv2.VideoCapture(0)
img = cv2.imread('color_balls_small.jpg')

#cv2.imshow('image',img)

while(True):
    ret,frame = cap.read()

    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cv2.waitKey(0)