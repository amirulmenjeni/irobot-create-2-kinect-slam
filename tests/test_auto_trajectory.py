import time
import slam
import cv2
from robot import Robot, StaticPlotter
import datetime as dt
import imdraw_util as imdraw
import math
import numpy as np
import config



try:
    r = Robot()
    plotter = StaticPlotter(2, [r.get_pose()[:2]] * 2, ['kx-', 'cx--'])

    print('RUN')
    r.drive(0, 0)
    time.sleep(2)

    # waypoints = [(100, 0), (100, 100), (0, 100)]
    # waypoints = [(100, -50), (0, -100), (100, -150), (0, -200)]
    waypoints = [(100, 0), (100, -100), (0, -100), (0, 0)]
    # waypoints = [(100, 0)]
    # waypoints = [(100, 0), (100, -100)]
    r.plotter.set_waypoints(waypoints)
    r.drive_trajectory(7, waypoints)

    # Wait for the robot autonomous driving flag turns true.
    while not r.is_autonomous:
        pass

    # Wait for the autonomous driving is complete.
    while r.is_autonomous:

        best_particle = r.fast_slam.highest_particle()

        d = slam.d3_map(best_particle.m, invert=True)
        imdraw.draw_robot(d, config.GRID_MAP_RESOLUTION,\
                r.get_pose(), radius=2)
        # imdraw.draw_vertical_line(d, config.GRID_MAP_SIZE[1] // 2, (0, 0, 255))
        # imdraw.draw_horizontal_line(d, config.GRID_MAP_SIZE[0] // 2, (0, 0, 255))

        try:
            random_pos = slam.cell_to_world_pos(r.random_cell,
                    config.GRID_MAP_SIZE, config.GRID_MAP_RESOLUTION,
                    center=True)
            imdraw.draw_vertical_line(d, random_pos[1], (0, 0, 255))
            imdraw.draw_horizontal_line(d, random_pos[0], (0, 0, 255))
            # imdraw.draw_square(d, config.GRID_MAP_RESOLUTION,\
            #         random_pos, (0, 0, 255), width=1)
        except:
            pass

        particles = r.fast_slam.particles
        for particle in particles:
            # print('particle, pose:', particle, particle.x)
            imdraw.draw_robot(d, config.GRID_MAP_RESOLUTION, particle.x,
                bgr=(0,255,0), radius=1, show_heading=True)
            # imdraw.draw_square(d, config.GRID_MAP_RESOLUTION, particle.x, 
            #     bgr=(0,255,0), width=1)

        cv2.imshow('map', d)
        cv2.waitKey(250)

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
