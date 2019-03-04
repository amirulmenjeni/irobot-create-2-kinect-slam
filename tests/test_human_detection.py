import cv2
import sys

image_path = sys.argv[1]

image = cv2.imread(image_path)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

hog_humans = hog.detectMultiScale(image, 0, (8, 8), (32, 32) , 1.05, 2, False)[0]

for (x, y, w, h) in hog_humans:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
