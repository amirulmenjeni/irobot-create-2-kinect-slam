##################################################
# DIRECTORIES.
##################################################

# Absolute root directory of this project.
PROJECT_PATH = '/home/amenji/git/fyp-robot-navigation/'

# The redist path for OpenNI2 and NiTE2 containing the relevant files.
PRIMESENSE_REDIST_PATH = PROJECT_PATH + '/redist/'

##################################################
# DISPLAY / UI.
##################################################
MAP_SCALE_FACTOR = 2

##################################################
# ROBOT SPECIFIC.
##################################################

# Axial length between the differential-drive robot's left and right wheels.
AXIAL_LENGTH = 26.0

# PID values.
PID_X = (0.15, 0, 0)
PID_Y = (0.15, 0, 0)

NORMAL_DRIVE_SPEED = 5
NORMAL_ROTATE_SPEED = 0.0873
ESCAPE_OBSTACLE_SPEED = 5

##################################################
# SLAM
##################################################
CONTROL_DELTA_TIME = 0.25
MEASURE_DELTA_TIME = 0
PF_NUM_PARTICLES = 120
GRID_MAP_RESOLUTION = 10.0
GRID_MAP_SIZE = (300, 300)
MOTION_NOISE = (1e-4, 1e-4, 1e-6, 1e-6)
OCCU_THRES = 0.90

# Kernel defining the shape representation of the robot projected on the grid
# map.
BODY_KERNEL_RADIUS = 1

##################################################
# SOUNDS
##################################################
SND_INIT = PROJECT_PATH + './resource/voice/m4_init.wav'
SND_GREET = PROJECT_PATH + './resource/voice/m4_greet.wav'
SND_APPROACH = PROJECT_PATH + './resource/voice/m4_approach_guest.wav'
SND_EXPLORE = PROJECT_PATH + './resource/voice/m4_explore.wav'
SND_SEE_HUMAN = PROJECT_PATH + './resource/voice/m4_see_human.wav'
SND_OOPS = PROJECT_PATH + './resource/voice/m4_oops.wav'
