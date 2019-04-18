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
import logging

r = Robot()

print('RUN')
r.drive(0, 0)
time.sleep(2)

map_image = None
save_img = None
        
logging.basicConfig(level=logging.INFO)

r.run()

try:

    # Wait for the autonomous driving is complete.
    while True:

        best_particle = r.fast_slam.highest_particle()
        estimated_pose = r.fast_slam.estimate_pose()

        entropy_map = slam.entropy_map(best_particle.m)

        save_img = best_particle.m

        map_image = slam.d3_map(best_particle.m, invert=True)
        ent_image = slam.d3_map(entropy_map)
        hum_image = slam.d3_map(r.hum_grid_map)

        imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION,\
                best_particle.x, radius=2)

        imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION,
                best_particle.x, bgr=(0, 255, 0), radius=1, show_heading=True)

        if r.goal_cell is not None:

            # Clearly show the goal cell.
            imdraw.draw_vertical_line(map_image, r.goal_cell[1], (0, 0, 255))
            imdraw.draw_horizontal_line(map_image, r.goal_cell[0], (0, 0, 255))

        if r.nearest_human is not None:

            # print('nearest_human:', r.nearest_human)
            # print('robot pos:', slam.world_to_cell_pos(estimated_pose[:2],\
            #     config.GRID_MAP_SIZE, config.GRID_MAP_RESOLUTION))

            imdraw.draw_square(map_image, config.GRID_MAP_RESOLUTION,\
                r.nearest_human, (0, 0, 255), pos_cell=True)

        # Draw particles of fastSLAM.
        # particles = r.fast_slam.particles
        # for particle in particles:
        #     imdraw.draw_robot(map_image, config.GRID_MAP_RESOLUTION, particle.x,
        #         bgr=(0,255,0), radius=1, show_heading=True)

        for cell in r.current_solution:
            imdraw.draw_square(map_image, config.GRID_MAP_RESOLUTION,\
                cell, (255, 0, 0), width=1, pos_cell=True)

        cv2.imshow('map', np.hstack((map_image, hum_image, ent_image)))

        cv2.waitKey(100)

except KeyboardInterrupt:
    # Save the map once done driving.
    print('Saving...')
    now = dt.datetime.now()
    year, month, day, hr, mn = now.year, now.month, now.day, now.hour,\
       now.minute
    save_status = np.save(\
        '../data/map_data/{0}_{1}_{2}_{3}_{4}.npy'.format(\
                year, month, day, hr, mn), save_img)
    print('save_status:', save_status)

    print('Stop explore...')
    r.drive(0, 0)
    time.sleep(0.5)
    r.clean_up()
