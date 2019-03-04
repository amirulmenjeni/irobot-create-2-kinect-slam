import test_kinect as tk
import numpy as np
import cv2

MAP_SIZE = (500, 500)
CELL_SIZE_CM = 10.0

m = np.zeros(MAP_SIZE)
m.fill(255)
m[MAP_SIZE[0] // 2, :] = 0
m[:, MAP_SIZE[1] // 2] = 0
cells = {}

w0 = (0, 0)
w1 = (100, 2000)

p0 = tk.world_to_cell_pos(w0, MAP_SIZE, CELL_SIZE_CM)
p1 = tk.world_to_cell_pos(w1, MAP_SIZE, CELL_SIZE_CM)
print('w0:', w0, 'w1:', w1)
print('p0:', p0, 'p1:', p1)
tk.draw_line(m, cells, p0, p1)

# m[p0] = 0
# m[p1] = 0

# cv2.imshow('hi', m)

# cv2.waitKey(0)


