import time
import slam
import cv2
from robot import Robot, StaticPlotter
import datetime as dt

r = Robot()

plotter = StaticPlotter(2, [r.get_pose()[:2]] * 2, ['kx-', 'cx--'])

try:
    print('RUN')
    r.drive(0, 0)
    time.sleep(2)

    # waypoints = [(100, 0), (100, 100), (0, 100)]
    # waypoints = [(100, -50), (0, -100), (100, -150), (0, -200)]
    waypoints = [(100, 0), (100, -100), (0, -100), (0, 0)]
    # waypoints = [(100, 0), (100, -100)]
    r.plotter.set_waypoints(waypoints)
    r.drive_trajectory(5.5, waypoints)

    # Wait for the robot autonomous driving flag turns true.
    while not r.is_autonomous:
        pass

    # Wait for the autonomous driving is complete.
    while r.is_autonomous:

        d = slam.d3_map(r.posterior)
        slam.draw_square(d, 10.0, r.get_pose(), (255, 0, 0), width=3)
        slam.draw_vertical_line(d, 250, (0, 0, 255))
        slam.draw_horizontal_line(d, 250, (0, 0, 255))

        cv2.imshow('map', d)
        cv2.waitKey(250)

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
