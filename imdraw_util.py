import numpy as np
import cv2
import slam
import rutil

def draw_square(d3_map, resolution, pos, bgr, width=3):

    assert width > 0
    assert resolution > 0

    map_size = d3_map.shape[:2]
    y0, x0 = slam.world_to_cell_pos(pos, map_size, resolution)

    if width > 1:
        r = np.round(width / 2).astype(int)
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):

                x = x0 + i
                y = y0 + j

                if x < 0 or x >= map_size[1] or\
                   y < 0 or y >= map_size[0]:
                    continue

                for k in range(3):
                    d3_map[y, x, k] = bgr[k]
    else:
        for k in range(3):
            d3_map[y0, x0, k] = bgr[k]

def draw_vertical_line(d3_map, x_pos, bgr):

    for k in range(3):
        d3_map[:, x_pos, k] = bgr[k]

def draw_horizontal_line(d3_map, y_pos, bgr):

    for k in range(3):
        d3_map[y_pos, :, k] = bgr[k]

def draw_robot(d3_map, resolution, pose, bgr=(255, 255, 0), radius=5,
        show_heading=True):

    map_size = d3_map.shape[:2]
    y0, x0 = slam.world_to_cell_pos(pose[:2], map_size, resolution)

    # Draw circle for body.
    cv2.circle(d3_map, (x0, y0), radius, bgr, thickness=-1)

    # Draw a line for heading.
    if show_heading:
        p0 = pose[:2]
        dd = np.array([100, 0])
        H = rutil.rigid_trans_mat3(np.array([0, 0, pose[2]]))
        p1 = p0 + rutil.transform_pts_2d(H, np.array([dd]))[0]
        y1, x1 = slam.world_to_cell_pos(p1, map_size, resolution)
        cv2.line(d3_map, (x0, y0), (x1, y1), bgr)
