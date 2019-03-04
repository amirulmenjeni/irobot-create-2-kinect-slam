# The resolution of the occupancy grid map in unit per pixel.
GRID_MAP_RESOLUTION = 10.0
GRID_MAP_SIZE = (300, 300)

# PID values.
PID_X = (0.15, 0, 0)
PID_Y = (0.15, 0, 0)

# Axial length between the differential-drive robot's left and right wheels.
AXIAL_LENGTH = 26.0

# SLAM
CONTROL_DELTA_TIME = 0.1
MEASURE_DELTA_TIME = 0
PF_NUM_PARTICLES = 150

# Human detection.
CASCADE_XML_PATH =\
'/home/amenji/git/fyp-robot-navigation/data/haarcascades/haarcascade_frontalface_default.xml'
