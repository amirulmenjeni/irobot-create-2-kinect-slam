import numpy as np
import slam
import cv2
import time
import rutil

def main():

    N_SENSE = 100
    sense = [0] * N_SENSE
    for i in range(N_SENSE):
        sense[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    MAP_SIZE = (500, 500)
    RESOLUTION = 5

    ogm_posterior = np.zeros(MAP_SIZE)

    rpose = np.array([0, 0, 0])

    it = 0
    while 1:

        curr_frame = sense[it]

        H = rutil.rigid_trans_mat3(rpose)
        curr_frame = rutil.transform_pts_2d(H, curr_frame)

        end_pt_cells = slam.world_frame_to_cell_pos(curr_frame, MAP_SIZE,
            RESOLUTION)

        slam.occupancy_grid_mapping(ogm_posterior, np.copy(ogm_posterior),
            rpose, end_pt_cells, RESOLUTION)

        particle_pose = np.array([10, 0, 0])
        prob = slam.map_matching_measurement_model(particle_pose, curr_frame,
                ogm_posterior, RESOLUTION)

        print('prob:', prob)

        d = slam.d3_map(ogm_posterior)
        slam.draw_vertical_line(d, 250, (0, 0, 255))
        slam.draw_horizontal_line(d, 250, (0, 0, 255))

        cv2.imshow('map', d)
        cv2.waitKey(250)

        it = (it + 1) % N_SENSE

if __name__ == '__main__':
    main()
