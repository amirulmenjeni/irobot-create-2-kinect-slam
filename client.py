import numpy as np
import socket
import config
import time
import datetime
import cv2
from pynput.keyboard import Key, Listener, KeyCode

HOST = '192.168.31.108'
PORT = 9000
MAP_SIZE = config.GRID_MAP_SIZE
RGB_SIZE = (120, 160)
BYTES_MAP = MAP_SIZE[0] * MAP_SIZE[1] * 3
SCALE_FACTOR = 2
AUTO_SAVE_INTERVAL = 60

DRIVE_KEYS = ['w', 'a', 's', 'd']

input_cell = (2**32-1, 2**32-1)
input_key = 0

KEY_F1 = 190
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

    print('input key:', input_key)

def save_map_image(map_image):

    now = datetime.datetime.now()
    year, month, day, hr, mn = now.year, now.month, now.day, now.hour,\
        now.minute

    filename = "./saves/{:04d}{:02d}{:02d}_{:02d}{:02d}.png".format(\
        year, month, day, hr, mn)

    img = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(filename, img)

    print(filename, 'saved.')

cv2.namedWindow('Display')
cv2.setMouseCallback('Display', on_mouse, None)

# Collect events.
listener = Listener(on_release=on_release)

print('Starting listener...')
listener.start()

print('Waiting listener to get ready...')
listener.wait()

print('Listener is now ready.')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    s.connect((HOST, PORT))
    f = config.MAP_SCALE_FACTOR

    cv2.imshow('Display', np.full((30, 30), 0))
    cv2.waitKey(10)

    try:

        tstart = time.time()
        is_conn = True
        while True:

            try:
                data = b''
                while len(data) != BYTES_MAP:
                    data += s.recv(4096)
                map_img = np.frombuffer(data, np.uint8)
                map_img = map_img.reshape(MAP_SIZE[1], MAP_SIZE[0], 3)

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

                if input_key == KEY_F1:
                    save_map_image(map_img)

                if (time.time() - tstart) >= AUTO_SAVE_INTERVAL:
                    save_map_image(map_img)
                    tstart = time.time()

                # Reset input.
                input_cell = 0
                input_key = 0

                new_w = map_img.shape[1] * SCALE_FACTOR
                new_h = map_img.shape[0] * SCALE_FACTOR
                map_img = cv2.resize(map_img, (new_h, new_w),
                    interpolation=cv2.INTER_AREA)

                cv2.imshow('Display', map_img)
                cv2.waitKey(100)

            except socket.error:
                print('Connection lost! Attempting to reconnect...')
                is_conn = False
                while not is_conn:
                    try:
                        s.connect((HOST, PORT))
                        is_conn = True
                        print('Reconnection successful!')
                    except socket.error:
                        print('Trying again...')
                        time.sleep(2)


    except KeyboardInterrupt:
        listener.join()
