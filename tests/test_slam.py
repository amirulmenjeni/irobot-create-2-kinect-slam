import time
import slam
import cv2
from robot import Robot

r = Robot()

try:
    r.drive(0, 0)

    waypoints = [(100, 0), (100, -100), (0, -100), (0, 0)]
    r.drive_trajectory(10, waypoints)

    while 1:
        d = slam.d3_map(r.posterior)
        slam.draw_square(d, 10.0, r.get_pose(), (255, 0, 0), r=3)
        slam.draw_vertical_line(d, 250, (0, 0, 255))
        slam.draw_horizontal_line(d, 250, (0, 0, 255))

        cv2.imshow('map', d)
        cv2.waitKey(250)

except KeyboardInterrupt:
    r.halt()
    r.clean_up()
