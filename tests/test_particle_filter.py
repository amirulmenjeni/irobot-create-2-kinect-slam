import numpy as np
import slam
import cv2
import time
import rutil
import imdraw_util as imdraw

def main():

    N_SENSE = 100
    sense = [0] * N_SENSE
    for i in range(N_SENSE):
        sense[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    MAP_SIZE = (500, 500)
    RESOLUTION = 5
    DELTA_TIME = 0.25

    ogm = np.full(MAP_SIZE, 0.5)

    rpose = np.array([0, 0, 0])

    pf = slam.ParticleFilter(DELTA_TIME, MAP_SIZE, RESOLUTION, num_particles=100)

    it = 0
    while 1:

        curr_frame = sense[it]

        control = np.array([5, 0])

        H = rutil.rigid_trans_mat3(rpose)
        curr_frame = rutil.transform_pts_2d(H, curr_frame)

        rpose[:2] = rpose[:2] + control

        pf.update(curr_frame, control, rpose, ogm)

        end_pt_cells = slam.world_frame_to_cell_pos(curr_frame, MAP_SIZE,
            RESOLUTION)

        slam.occupancy_grid_mapping(ogm, np.copy(ogm), rpose, end_pt_cells,
                RESOLUTION)

        d = slam.d3_map(ogm, invert=True)

        for p in pf.particles:
            imdraw.draw_robot(d, RESOLUTION, p.x)

        cv2.imshow('map', d)
        cv2.waitKey(250)

        it = (it + 1) % N_SENSE

if __name__ == '__main__':
    main()
