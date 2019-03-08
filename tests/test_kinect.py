import sys
sys.path.append('../')

import cv2
import math
import numpy as np
import time
import freenect
import slam
import config
import imdraw_util as imdraw

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

def dilate_map():

    image = cv2.imread('../map_images/2019_2_14_22_50.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    ret, thres = cv2.threshold(blur, 255*0.3, 255, cv2.THRESH_BINARY_INV)

    # dilate = cv2.morphologyEx(thres, cv2.MORPH_OPEN, (10, 10), iterations=5)
    erode = cv2.dilate(thres, (7,7), iterations=5)

    new_image = np.copy(image)
    new_image[erode == 255] = 0

    cv2.imshow('image', np.hstack((gray, thres, erode)))
    cv2.imshow('new', np.hstack((image, new_image)))

    while 1:
        if cv2.waitKey(0) == ord('q'):
            break

def test():

    xy_humans = np.load('xy_humans.npy')

    print(xy_humans)

    xy_humans = slam.world_frame_to_cell_pos(xy_humans,
        config.GRID_MAP_SIZE, config.GRID_MAP_RESOLUTION)

    print(xy_humans)

    map_image = np.full(config.GRID_MAP_SIZE, 0.5)
    slam.update_human_grid_map(map_image, xy_humans)

    cv2.GaussianBlur(map_image, (5, 5), 1, map_image, 1)

    map_image = slam.d3_map(map_image)

    imdraw.draw_horizontal_line(map_image, 150, (0, 0, 255))
    imdraw.draw_vertical_line(map_image, 150, (0, 0, 255))

    cv2.imshow('image', map_image)

    while 1:
        if cv2.waitKey(0) == ord('q'):
            break

dilate_map()
