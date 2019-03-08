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

print('RUN')
r.drive(0, 0)
time.sleep(2)

map_image = None

try:

    while not r.is_autonomous:
        pass

    # Wait for the autonomous driving is complete.
    while r.is_autonomous:

        best_particle = r.fast_slam.highest_particle()
        estimated_pose = r.fast_slam.estimate_pose()

        entropy_map = slam.entropy_map(best_particle.m)

        map_image = slam.d3_map(best_particle.m, invert=True)
        ent_image = slam.d3_map(entropy_map)
        hum_image = slam.d3_map(r.hum_grid_map)

        imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION,\
                best_particle.x, radius=2)

        imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION,
                best_particle.x, bgr=(0, 255, 0), radius=1, show_heading=True)

        if estimated_pose is not None:
            imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION,\
                    estimated_pose, bgr=(153,51,255), radius=2)

        if r.goal_cell is not None:

            # Clearly show the goal cell.
            imdraw.draw_vertical_line(map_image, r.goal_cell[1], (0, 0, 255))
            imdraw.draw_horizontal_line(map_image, r.goal_cell[0], (0, 0, 255))

        # Draw particles of fastSLAM.
        # particles = r.fast_slam.particles
        # for particle in particles:
        #     imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION, particle.x,
        #         bgr=(0,255,0), radius=1, show_heading=True)

        # for cell in r.path:
        #     imdraw.draw_square(map_image, config.GRID_MAP_RESOLUTION,\
        #         cell, (125, 255, 125), width=1, pos_cell=True)

        cv2.imshow('map', np.hstack((map_image, hum_image, ent_image)))

        cv2.waitKey(100)

    # Save the map once done driving.
    print('Saving...')
    now = dt.datetime.now()
    year, month, day, hr, mn = now.year, now.month, now.day, now.hour,\
       now.minute
    save_status = cv2.imwrite('./{0}_{1}_{2}_{3}_{4}.jpg'.format(\
        year, month, day, hr, mn), d)
    print('save_status:', save_status)

except KeyboardInterrupt:
    print('Stop explore...')
    r.drive(0, 0)
    time.sleep(0.5)
    r.clean_up()
