from kinematic import Trajectory

# j = Trajectory(1, (0, 0), 90, (0, 10))
# j.add_waypoint(10, 10)
# j.add_waypoint(10, 0)

j = Trajectory(10, (0, 0), 0, (100, 0))
j.add_waypoint(100, -100)
j.add_waypoint(0, -100)
j.add_waypoint(0, 0)

# j.plot_motion()
j.plot_trajectory()
