import sys
sys.path.append('../')

import argparse
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

parser = argparse.ArgumentParser(description='Start robot.')
parser.add_argument('--show-display', help='Show the map display, only work if
        this script is run locally (e.g., not over ssh connection).',\
        default=True, action='store_true')
parser.add_argument('--disable-auto', help='Disable autonomous driving.',\
        default=False, action='store_true')
parser.add_argument('--usb-port', help='The USB port to the roomba robot.',\
        default='/dev/ttyUSB0')
args = parser.parse_args()

r = Robot()
r.drive_velocity(0, 0)
r.run(show_display=args.show_display, disable_auto=args.disable_auto)
