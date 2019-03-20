import numpy as np
import socket
import config
import cv2

HOST = '192.168.31.105'
PORT = 9000
MAP_SIZE = config.GRID_MAP_SIZE
BYTES_MAP = MAP_SIZE[0] * MAP_SIZE[1] * 3

input_cell = 0

def on_mouse(event, x, y, flags, param):

    global input_cell
    
    if event == cv2.EVENT_LBUTTONUP:
        print('Clicked:', (y, x))
        input_cell = (y, x) 

cv2.namedWindow('Display')
cv2.setMouseCallback('Display', on_mouse, None)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    s.connect((HOST, PORT))

    while True:

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
        s.sendall(snd)

        # Reset input.
        input_cell = 0

        cv2.imshow('Display', map_img)

        cv2.waitKey(100)
