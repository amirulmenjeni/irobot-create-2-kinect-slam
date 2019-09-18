import numpy as np
import socket
import config
import time
import datetime
import cv2
import argparse
import re
import rutil
from pynput.keyboard import Key, Listener, KeyCode

MAP_SIZE = config.GRID_MAP_SIZE
RGB_SIZE = (120, 160)
BYTES_MAP = MAP_SIZE[0] * MAP_SIZE[1] * 3
BYTES_HUM = MAP_SIZE[0] * MAP_SIZE[1] * 3
SCALE_FACTOR = 2
AUTO_SAVE_INTERVAL = 60

# Window names.
GRID_MAP_WINDOW = 'GRID_MAP_WINDOW'
HUMAN_MAP_WINDOW = 'HUMAN_MAP_WINDOW'

DRIVE_KEYS = ['w', 'a', 's', 'd']

input_cell = (2**32-1, 2**32-1)
input_key = 0

KEY_F1 = 190
KEY_F2 = 191
KEY_SPACE = 32
KEY_ESC = 27

def on_mouse(event, x, y, flags, param):

    global input_cell

    if event == cv2.EVENT_LBUTTONUP:
        print('Clicked:', (y, x))
        input_cell = (y // SCALE_FACTOR, x // SCALE_FACTOR) 

def on_release(key):
    
    global input_key

    print('key:', key, type(key))

    if type(key) == KeyCode:
        if key.char in DRIVE_KEYS:
            input_key = ord(key.char)

    elif type(key) == Key:
        if key == Key.space:
            input_key = KEY_SPACE
        elif key == Key.esc:
            input_key = KEY_ESC
        elif key == Key.f1:
            input_key = KEY_F1
        elif key == Key.f2:
            input_key = KEY_F2

    print('input key:', input_key)

    if input_key == KEY_ESC:
        return False

def save_map_image(map_image):

    now = datetime.datetime.now()
    year, month, day, hr, mn = now.year, now.month, now.day, now.hour,\
        now.minute

    filename = "./saves/{:04d}{:02d}{:02d}_{:02d}{:02d}.png".format(\
        year, month, day, hr, mn)

    img = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(filename, img)

    print(filename, 'saved.')

def on_exit():
    print('Saving map data...')
    np.save('./saves/' + rutil.now_file_name(postfix='.npy'), frame_array)
    listener.join()
    print('Exiting...')

##################################################
# Define arguments.
##################################################

parser = argparse.ArgumentParser(description='Client program for '\
        'interacting with the on-line 2D map generated using SLAM.')
parser.add_argument('-a', '--ipv4', type=str,\
        default='127.0.0.1', help='IPv4 address of the host.')
parser.add_argument('-p', '--port', type=int,\
        default='9000', help='The port of the address to connect.')
parser.add_argument('-S', '--save-map-display', default=False,\
        action='store_true',\
        help='Save the map generation display during the current session as '\
            'a numpy array.')

args = parser.parse_args()

##################################################
# Initialize.
##################################################

# Set the named window.
cv2.namedWindow(GRID_MAP_WINDOW)
cv2.setMouseCallback(GRID_MAP_WINDOW, on_mouse, None)

# Collect events.
listener = Listener(on_release=on_release)

# Save frames of the map display.
frame_array = []

##################################################
# Start.
##################################################

print('Starting listener...')
listener.start()

print('Waiting listener to get ready...')
listener.wait()

print('Listener is now ready.')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    s.connect((args.ipv4, args.port))
    f = config.MAP_SCALE_FACTOR

    # cv2.imshow(GRID_MAP_WINDOW, np.full((30, 30), 0))
    # cv2.waitKey(10)

    try:

        tstart = time.time()
        is_conn = True
        while True:

            try:
                data = b''
                while len(data) != (BYTES_MAP + BYTES_HUM):
                    data += s.recv(4096)

                # Receive array buffer containing the map images.
                map_buf = np.frombuffer(data, np.uint8)[:BYTES_MAP]
                map_img = map_buf.reshape(MAP_SIZE[1], MAP_SIZE[0], 3)

                hum_buf = np.frombuffer(\
                        data, np.uint8)[BYTES_MAP:BYTES_MAP+BYTES_HUM]
                hum_img = hum_buf.reshape(MAP_SIZE[1], MAP_SIZE[0], 3)

                # Send input data.
                snd = 0
                if input_cell != 0:
                    y, x = input_cell
                    y_buff = y.to_bytes(4, byteorder='big')
                    x_buff = x.to_bytes(4, byteorder='big')
                    snd = y_buff + x_buff
                else:
                    snd = bytes([255 for _ in range(8)])
                snd += (input_key).to_bytes(4, byteorder='big')

                s.sendall(snd)

                if input_key == KEY_ESC:
                    break

                if input_key == KEY_F1:
                    save_map_image(map_img)

                if (time.time() - tstart) >= AUTO_SAVE_INTERVAL:
                    save_map_image(map_img)
                    tstart = time.time()

                if args.save_map_display:
                    frame_array.append(map_img)

                # Reset input.
                input_cell = 0
                input_key = 0

                new_w = map_img.shape[1] * SCALE_FACTOR
                new_h = map_img.shape[0] * SCALE_FACTOR
                map_img = cv2.resize(map_img, (new_h, new_w),
                    interpolation=cv2.INTER_AREA)
                hum_img = cv2.resize(hum_img, (new_h, new_w),
                    interpolation=cv2.INTER_AREA)

                cv2.imshow(GRID_MAP_WINDOW, map_img)
                cv2.imshow(HUMAN_MAP_WINDOW, hum_img)
                cv2.waitKey(100)

            except socket.error:
                print('Connection lost! Attempting to reconnect...')
                is_conn = False
                while not is_conn:
                    try:
                        s.connect((args.ipv4, args.port))
                        is_conn = True
                        print('Reconnection successful!')
                    except socket.error:
                        print('Trying again...')
                        time.sleep(2)

    except KeyboardInterrupt:
        pass

on_exit()
