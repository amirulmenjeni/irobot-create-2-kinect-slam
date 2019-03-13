import sys
sys.path.append('../')

import time
import slam
import cv2
from robot import Robot
import datetime as dt
import imdraw_util as imdraw
import math
import numpy as np
import config
import logging

r = Robot()
r.drive_velocity(0, 0)
r.run(show_display=True)
