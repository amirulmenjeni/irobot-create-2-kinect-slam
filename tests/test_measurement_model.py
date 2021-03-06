import sys
sys.path.append('../')

import numpy as np
import math
import cv2
import time
import freenect 
import rutil
import imdraw_util as imdraw
import slam
from scipy import ndimage

def main():

    N_SENSE = 100
    sense = [0] * N_SENSE
    for i in range(N_SENSE):
        sense[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    MAP_SIZE = (500, 500)
    RESOLUTION = 10

    ogm_posterior = np.full(MAP_SIZE, 0.5)

    rpose = np.array([0, 0, 0])

    x_pos = 0

    it = 0
    while 1:

        curr_frame = sense[it]

        # J = rutil.rigid_trans_mat3([250, -250, 0], grids=True)
        # H = rutil.rigid_trans_mat3(rpose, True)
        # K = rutil.rigid_trans_mat3([-250, 250, 0], grids=True)
        # H = K @ H @ J
        # curr_frame = rutil.transform_pts_2d(H, curr_frame)

        end_pt_cells = slam.world_frame_to_cell_pos(curr_frame, MAP_SIZE,
            RESOLUTION)
        slam.occupancy_grid_mapping(ogm_posterior, np.copy(ogm_posterior),
            rpose, end_pt_cells, RESOLUTION)

        origin_cell = slam.world_to_cell_pos((0, 0), ogm_posterior.shape,\
                RESOLUTION)
        obs_cells, obs_dict = slam.observed_cells(origin_cell, end_pt_cells)

        obs_map = slam.observation_map(obs_dict, ogm_posterior.shape)

        # frame = rutil.transform_map(rpose, obs_map)
        frame = obs_map

        mask_free = cv2.inRange(frame, 0.00, 0.30)
        mask_occu = cv2.inRange(frame, 0.75, 1.00)
        mask = cv2.bitwise_or(mask_free, mask_occu)
        mask = rutil.transform_map(rpose, mask)

        mask2 = slam.observation_mask(obs_map)

        masked = cv2.bitwise_and(ogm_posterior, ogm_posterior, mask=mask)
        masked = rutil.transform_map(-rpose, masked)

        # imdraw.draw_robot(d, RESOLUTION, particle_pose)

        gmap = masked[masked > 0.0001]
        lmap = frame[abs(frame - 0.5) > 0.0001]

        print('gmap:', len(gmap))
        print('lmap:', len(lmap))

        n = len(lmap)
        m = np.sum(lmap + gmap) / (2*n)

        a = np.sum((gmap - m) * (lmap - m))
        b = np.sqrt(np.sum((gmap - m)**2) * np.sum((lmap - m)**2))

        prob = max(0, a/b)

        print('n:', n)
        print('m:', m)
        print('a:', a)
        print('b:', b)
        print('a/b:', a/b)
        print('prob:', prob)

        # maskedd3 = slam.d3_map(masked)
        # imdraw.draw_vertical_line(maskedd3, 250, (0, 0, 255))
        # ogmd3 = slam.d3_map(ogm_posterior)
        # imdraw.draw_vertical_line(ogmd3, 250, (0, 0, 255))
        # maskd3 = slam.d3_map(mask)

        # cv2.imshow('map', np.vstack((ogmd3, maskedd3, maskd3)))
        cv2.imshow('map', np.hstack((mask, mask2)))
        cv2.waitKey(250)

        it = (it + 1) % N_SENSE
        x_pos += 10

def correlation_test():

    N_SENSE = 100
    sense = [0] * N_SENSE
    for i in range(N_SENSE):
        sense[i] = np.load('./data/kinect_map/{0}.npy'.format(i))

    MAP_SIZE = (500, 500)
    RESOLUTION = 10

    grid_map = np.full(MAP_SIZE, 0.5)

    rpose = np.array([0, 0, 0])

    x_pos = 0

    it = 0
    while 1:

        curr_frame = sense[it]

        end_pt_cells = slam.world_frame_to_cell_pos(curr_frame, MAP_SIZE,
            RESOLUTION)
        slam.occupancy_grid_mapping(grid_map, np.copy(grid_map),
                rpose, end_pt_cells, RESOLUTION)

        origin_cell = slam.world_to_cell_pos((0, 0), grid_map.shape,\
                RESOLUTION)
        obs_cells, cell_dict = slam.observed_cells(origin_cell,\
                end_pt_cells)

        obsr_mat = slam.observation_matrix(cell_dict)

        tstart = time.time()
        for _ in range(60):
            prob = slam.correlation_measurement_model(obsr_mat, rpose, grid_map,
                    RESOLUTION)
            print('  prob:', prob)
        print('time:', time.time() - tstart)

        cv2.imshow('map', grid_map)
        cv2.waitKey(250)

        it = (it + 1) % 3

def save_depths():

    it = 0
    while 1:

        depth, _ = freenect.sync_get_depth()

        np.save('./data/kinect_depth/{0}.npy'.format(it), depth)
        print('iter:', it)

        time.sleep(0.25)

        it = it + 1
    
def likelihood_test():

    N_SENSE = 100
    depth = [0] * N_SENSE
    for i in range(N_SENSE):
        depth[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    MAP_SIZE = (300, 300)
    RESOLUTION = 10
    SLICE_ROW = 315

    grid_map = np.full(MAP_SIZE, 0.5)

    ppose = np.array([0, 0, 0])
    rpose = np.array([0, 0, 0])

    x_pos = 0

    it = 0
    while 1:


        # depth_slice = depth[it][SLICE_ROW,:]
        # depth_slice = depth_slice / 10 # mm to cm.

        # print(depth_slice)
        # xy_data, depth_data = slam.get_kinect_xy(SLICE_ROW)

        xy_data = depth[it]

        # print('xy_data.shape:', xy_data.shape)
        # print('depth_data.shape:', depth_data.shape)

        end_pt_cells = slam.world_frame_to_cell_pos(xy_data, MAP_SIZE,
                RESOLUTION)

        origin_cell = slam.world_to_cell_pos((0, 0), grid_map.shape,\
                RESOLUTION)
        obs_cells, obs_dict = slam.observed_cells(origin_cell, end_pt_cells)

        obs_mat = slam.observation_matrix(obs_dict, occu=0.9, free=-0.7)

        H = rutil.rigid_trans_mat3([0, 0, 0])
        obs_mat[:,:2] =\
                rutil.transform_cells_2d(H, obs_mat[:,:2], MAP_SIZE, RESOLUTION)

        tstart = time.time()
        for _ in range(150):
            slam.occupancy_grid_mapping2(grid_map, obs_mat)
        print('time:', time.time() - tstart)

        occ_cells = np.argwhere(grid_map > 0.75)

        prob = slam.likelihood_field_measurement_model(obs_mat[:,:2], ppose,
                occ_cells, MAP_SIZE, RESOLUTION)
        print('prob:', prob)

        d3 = slam.d3_map(grid_map)
        imdraw.draw_robot(d3, RESOLUTION, rpose, bgr=(255, 0, 0), radius=2)
        imdraw.draw_robot(d3, RESOLUTION, ppose, bgr=(0, 255, 0), radius=2)

        cv2.imshow('map', d3)
        cv2.waitKey(250)

        it = (it + 1) % 2

if __name__ == '__main__':
    # correlation_test() 
    likelihood_test()
    # save_depths()
    
