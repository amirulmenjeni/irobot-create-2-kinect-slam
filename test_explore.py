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

        d = slam.d3_map(best_particle.m, invert=True)
        imdraw.draw_robot(d, config.GRID_MAP_RESOLUTION,\
                r.get_pose(), radius=2)

        try:

            # The position of the random goal cell.
            random_pos = slam.cell_to_world_pos(r.random_cell,
                    config.GRID_MAP_SIZE, config.GRID_MAP_RESOLUTION,
                    center=True)

            # Clearly show the goal cell.
            imdraw.draw_vertical_line(d, random_pos[1], (0, 0, 255))
            imdraw.draw_horizontal_line(d, random_pos[0], (0, 0, 255))

        except:
            # Happens when the random cell is not yet generated.
            pass

        # Draw particles of fastSLAM.
        particles = r.fast_slam.particles
        for particle in particles:
            imdraw.draw_robot(d, config.GRID_MAP_RESOLUTION, particle.x,
                bgr=(0,255,0), radius=1, show_heading=True)

        cv2.imshow('map', d)
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

    # Plot the displacement of the robot.
    print('Drawing...')
    r.plotter.draw()

    r.halt()
    print('Halted')

except KeyboardInterrupt:
    r.halt()
    r.plotter.draw()
    r.clean_up()
