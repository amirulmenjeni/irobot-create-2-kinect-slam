##################################################
# DIRECTORIES.
##################################################

# Absolute root directory of this project.
PROJECT_PATH = '/home/amenji/git/fyp-robot-navigation/'

# The redist path for OpenNI2 and NiTE2 containing the relevant files.
PRIMESENSE_REDIST_PATH = PROJECT_PATH + '/redist/'

##################################################
# ROBOT SPECIFIC.
##################################################

# Axial length between the differential-drive robot's left and right wheels.
AXIAL_LENGTH = 26.0

# PID values.
PID_X = (0.15, 0, 0)
PID_Y = (0.15, 0, 0)

NORMAL_DRIVE_SPEED = 5
ESCAPE_OBSTACLE_SPEED = 5

##################################################
# SLAM
##################################################
CONTROL_DELTA_TIME = 0.1
MEASURE_DELTA_TIME = 0
PF_NUM_PARTICLES = 150
GRID_MAP_RESOLUTION = 10.0
GRID_MAP_SIZE = (300, 300)

##################################################
# SOUNDS
##################################################
SND_GREET = PROJECT_PATH + './resource/voice/m4_greet.wav'
SND_SEE_HUMAN = PROJECT_PATH + './resource/voice/m4_see_human.wav'
