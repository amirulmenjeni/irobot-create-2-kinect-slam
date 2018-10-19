import time
import math
from robot import Robot, AnimPlotter, StaticPlotter
from kinematic import Trajectory

r = Robot()

plotter = StaticPlotter(2, [r.get_pose()[:2]] * 2, ['kx-', 'cx--'])

try:
    print('RUN')
    r.drive(0, 0)
    time.sleep(2)

    # waypoints = [(100, 0), (100, 100), (0, 100)]
    # waypoints = [(100, -50), (0, -100), (100, -150), (0, -200)]
    waypoints = [(100, 0), (100, -100), (0, -100), (0, 0)]
    r.drive_trajectory(10, waypoints)


    # Wait for the robot autonomous driving flag turns true.
    while not r.is_autonomous:
        pass

    # Wait for the autonomous driving is complete.
    while r.is_autonomous:
        time.sleep(0.5)
        pass

    # Plot the displacement of the robot.
    r.plotter.draw()

    r.halt()
    print('Halted')
    plotter.draw()

except KeyboardInterrupt:
    r.halt()
    r.clean_up()
