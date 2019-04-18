import sys
sys.path.append('../')

import numpy as np
import cv2
import slam
import math
import config
import rutil
import imdraw_util as imdraw

def main():

    """
    Demo of probabilistic motion model.
    """

    MAP_SIZE = config.GRID_MAP_SIZE
    MAP_RESO = config.GRID_MAP_RESOLUTION

    map_image = np.full(MAP_SIZE, 0.5)

    beg_cell = (150, 150)
    end_cell = (150, 250)

    solution = slam.shortest_path(beg_cell, end_cell, map_image, 0.75)
    i = 0

    num_particles = 120

    x = np.array([0, 0, 0])
    w = 1 / num_particles
    particles =\
            [slam.Particle(x, w) for _ in range(num_particles)]

    NOISE = (1e-4, 1e-4, 1e-5, 1e-5)

    d3 = slam.d3_map(map_image, invert=True)

    flag_stop = False
    distance = 0
    while not flag_stop:

        if distance >= MAP_RESO * MAP_SIZE[1] // 2 - 50:
            break

        u_t = np.array([5, 0])
        distance += 5

        for particle in particles:
            x_t = slam.sample_motion_model_odometry(u_t, particle.x, NOISE)
            particle.x = np.copy(x_t)
            # imdraw.draw_robot(d3, MAP_RESO, particle.x,
            #     color=(0, 0, 0), border_thickness=1)

        a = np.array([p.x[:2] for p in particles])

        H, edges = np.histogramdd(a, 
            bins=(MAP_SIZE[1], MAP_SIZE[0]),
            range=([(-MAP_RESO * MAP_SIZE[1] // 2, MAP_RESO * MAP_SIZE[1] // 2),
                    (-MAP_RESO * MAP_SIZE[0] // 2, MAP_RESO * MAP_SIZE[0] // 2)]))

        tmp = H[H>0]
        min_count = np.min(tmp)
        max_count = np.max(tmp)

        for x in range(H.shape[1]):
            for y in range(H.shape[0]):
                if H[x, y] > 0:
                    grid_gray = d3[y, x][0]
                    wpos = slam.cell_to_world_pos((y, x), MAP_SIZE, MAP_RESO)
                    norm = rutil.min_max_normalize(H[x, y],
                            min_count, max_count)
                    if math.isnan(norm):
                        norm = 1.0
                    gray = 255 * norm
                    gray = (grid_gray + gray) / 2
                    gray = int(gray)
                    gray = 255 - gray
                    color = (gray, gray, gray)
                    # imdraw.draw_robot(d3, MAP_RESO, wpos, bgr=color, radius=1,
                    #         show_heading=False)
                    imdraw.draw_square(d3, MAP_RESO, wpos, color, width=1)

        imdraw.draw_grids(d3, 50, bgr=(0, 255, 0))

        cv2.imshow('Map', d3)

        cv2.waitKey(10)

    while True:
        if cv2.waitKey(10) == ord('q'):
            break

if __name__ == '__main__':
    main()
