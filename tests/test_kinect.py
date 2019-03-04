import cv2
import math
import numpy as np
import time
import freenect

def hog_default_people_detector():

    while 1:

        tstart = time.time()

        image, _ = freenect.sync_get_video()

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

        hog_humans =\
            hog.detectMultiScale(image, 0, (4, 4), (16, 16), 1.05, 2, False)[0]

        for (x, y, w, h) in hog_humans:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        print('time:', time.time() - tstart)

        cv2.imshow('image', image)
        cv2.waitKey(100)

def haar_classifier(casc_path):

    while 1:

        image, _ = freenect.sync_get_video()

        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(casc_path)
        body_rect = cascade.detectMultiScale(gray_img, scaleFactor=1.05,\
            minNeighbors=2, minSize=(30,30))

        if len(body_rect) <= 0:
            pass

        else:
            for (x, y, w, h) in body_rect:
                print('size:', (w, h))
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

        cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)

def kinect_human_regions():

    image, _ = freenect.sync_get_video()

    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    cascade =\
    cv2.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_default.xml')

    regions = cascade.detectMultiScale(gray_img, scaleFactor=1.05,\
        minNeighbors=2, minSize=(30,30))

    return regions

while 1:
    print(kinect_human_regions())

    cv2.waitKey(100)
# haar_classifier('../data/haarcascades/haarcascade_frontalface_default.xml')
# hog_default_people_detector()
