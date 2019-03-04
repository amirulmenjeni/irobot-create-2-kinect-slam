import sys
sys.path.append('../')

import time
import slam
import cv2
from robot import Robot, StaticPlotter
import datetime as dt
import imdraw_util as imdraw
import math
import numpy as np
import config

r = Robot()

try:

    print('RUN')
    r.drive(0, 0)
    time.sleep(2)

    while not r.is_autonomous:
        pass

    # Wait for the autonomous driving is complete.
    while r.is_autonomous:

        best_particle = r.fast_slam.highest_particle()
        estimated_pose = r.fast_slam.estimate_pose()

        entropy_map = slam.entropy_map(best_particle.m)

        map_image = slam.d3_map(best_particle.m, invert=True)
        ent_image = slam.d3_map(entropy_map)
        hum_image = slam.d3_map(r.hum_grid_map, invert=True)

        imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION,\
                r.get_pose(), radius=2)

        if estimated_pose is not None:
            imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION,\
                    estimated_pose, bgr=(153,51,255), radius=2)

        if r.goal_cell is not None:

            # Clearly show the goal cell.
            imdraw.draw_vertical_line(map_image, r.goal_cell[1], (0, 0, 255))
            imdraw.draw_horizontal_line(map_image, r.goal_cell[0], (0, 0, 255))

        # Draw particles of fastSLAM.
        particles = r.fast_slam.particles
        for particle in particles:
            imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION, particle.x,
                bgr=(0,255,0), radius=1, show_heading=True)

        cv2.imshow('map', np.hstack((map_image, hum_image, ent_image)))

        if r.camera_image is not None:
            cv2.imshow('camera', r.camera_image)
        cv2.waitKey(100)

    while 1:
        # Wait KeyboardInterrupt.
        pass

    # Save the map once done driving.
    now = dt.datetime.now()
    year, month, day, hr, mn = now.year, now.month, now.day, now.hour,\
       now.minute
    save_status = cv2.imwrite('./map_images/{0}_{1}_{2}_{3}_{4}.jpg'.format(\
        year, month, day, hr, mn), d)

    print('save_status:', save_status)

except KeyboardInterrupt:
    r.drive(0, 0)
    r.clean_up()
