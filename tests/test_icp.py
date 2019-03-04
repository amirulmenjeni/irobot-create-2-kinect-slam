import slam
import math
import cv2
import time
import numpy as np
import sys
from icp import icp

def gen_data_cloud(size, low, high):

    return (high - low) * np.random.random_sample((size, 2)) + low 

def do_transform(data, angle, tr):

    angle = np.radians(angle)

    t = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)]
    ])

    return np.dot(t, data.T).T + np.array(tr)

def random_transform(data, low_angle, high_angle):

    low_angle = np.radians(low_angle)
    high_angle = np.radians(high_angle)

    angle = (high_angle - low_angle) * np.random.rand() + low_angle

    t = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)]
    ])

    return np.dot(t, data.T).T

def get_grids(cells, frame, pos, map_size, resolution):

    occ_cells = slam.world_frame_to_cell_pos(frame, map_size, resolution)

    x1, y1 = slam.world_to_cell_pos(pos, map_size, resolution)

    for x0, y0 in occ_cells:
        slam.__beam_line(cells, (x1, y1), (x0, y0))

def keys_to_numpy_arr(dic):

    return np.array(list(dic.keys()))

def test_icp(src, dst, rot=0, tr=(0, 0), init_tr=[0, 0, 0], maxiter=50):

    mse_0 = np.inf
    mse_1 = np.inf

    occ_min_sz = min(src.shape[0], dst.shape[0])

    rot = np.radians(rot)

    rot_mat = np.array([
        [np.cos(rot), -np.sin(rot)],
        [np.sin(rot),  np.cos(rot)]
    ])

    init_dx, init_dy, init_da = init_tr
    init_da = np.radians(init_da)
    init_tr = np.array([
        [np.cos(init_da), -np.sin(init_da), init_dx],
        [np.sin(init_da),  np.cos(init_da), init_dy],
        [0              ,  0              , 1      ]
    ])

    src = np.copy(src)
    dst = np.copy(dst)

    print('src points: ({0})\n'.format(src.shape[0]), src)
    print('dst points: ({0})\n'.format(dst.shape[0]), dst)

    src = np.dot(rot_mat, src.T).T + tr

    print('src points: ({0})\n'.format(src.shape[0]), src)
    print('dst points: ({0})\n'.format(dst.shape[0]), dst)

    pos = [0, 0]
    map_size = (500, 500)
    resolution = 10.0

    src_occ = slam.world_frame_to_cell_pos(src, map_size, resolution)
    dst_occ = slam.world_frame_to_cell_pos(dst, map_size, resolution)

    print('src occ: ({0})\n'.format(src_occ.shape[0]), src_occ)
    print('dst occ: ({0})\n'.format(dst_occ.shape[0]), dst_occ)

    prev_cells = {}
    curr_cells = {}

    get_grids(curr_cells, src, pos, map_size, resolution)
    get_grids(prev_cells, dst, pos, map_size, resolution)

    src = keys_to_numpy_arr(curr_cells)
    dst = keys_to_numpy_arr(prev_cells)

    min_sz = min(src.shape[0], dst.shape[0])
    src = src[:min_sz]
    dst = dst[:min_sz]

    mse_0 = ((src_occ[:occ_min_sz] - dst_occ[:occ_min_sz]) ** 2).mean()

    print('src cells: ({0})\n'.format(src.shape[0]), src)
    print('dst cells: ({0})\n'.format(dst.shape[0]), dst)

    transformation = icp.icp(src, dst, init_pose=init_tr,
        max_iterations=maxiter)[0]

    print('Transform matrix, T:\n', transformation)

    rt = transformation[:2,:2]
    tt = transformation[:2,2].T

    src_occ = np.dot(rt, src_occ.T).T + tt
    src_occ = src_occ.round(0).astype(int)

    mse_1 = ((src_occ[:occ_min_sz] - dst_occ[:occ_min_sz]) ** 2).mean()

    print('Apply T to src_occ ({0}):\n'.format(src_occ.shape[0]), src_occ)

    print('MSEs:')
    print('  mse_0:', mse_0)
    print('  mse_1:', mse_1)
    print('  error:', 1 - ((mse_0 - mse_1) / mse_0))

def main():

    posterior = np.zeros((500, 500))

    sense = [0] * 100
    for i in range(len(sense)):
        sense[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    prev_frame = None
    curr_frame = None

    prev_cells = None

    it = 0

    scan_pose = np.array([0, 0, 0])

    while 1:

        curr_frame = sense[it]

        tr = (0, 0)
        angle = np.random.randint(-5, 5)
        scan_pose[2] = angle
        # curr_frame = do_transform(curr_frame, angle, tr)

        curr_grids = slam.world_frame_to_cell_pos(curr_frame, posterior.shape,
            10.0)

        x1, y1 = slam.world_to_cell_pos(scan_pose[:2], posterior.shape,
            10.0)

        curr_cells = {}
        for x0, y0 in curr_grids:
            slam.beam_line(curr_cells, (x1, y1), (x0, y0))

        if prev_cells is not None:

            src = np.array(list(curr_cells.keys()))
            dst = np.array(list(prev_cells.keys()))

            min_sz = min(len(src), len(dst))
            src = src[:min_sz]
            dst = dst[:min_sz]

            theta = np.radians(angle)
            init_tr = np.array([
                [np.cos(theta), -np.sin(theta), tr[0]],
                [np.sin(theta),  np.cos(theta), tr[1]],
                [0            ,  0            , 1    ]
            ])

            transformation = icp.icp(src, dst, init_pose=init_tr)[0]

            rt = transformation[:2,:2]
            tt = transformation[:2,2].T

            curr_grids = np.dot(rt, curr_grids.T).T + tt
            curr_grids = curr_grids.round(0).astype(int)
            print(rt, tt)

        d = slam.d3_map(posterior)
        slam.draw_vertical_line(d, 250, (0, 0, 255))
        slam.draw_horizontal_line(d, 250, (0, 0, 255))

        # occ_grids = slam.world_frame_to_cell_pos(curr_frame, posterior.shape,
        #         10.0)

        slam.occupancy_grid_mapping(posterior,\
                np.copy(posterior), scan_pose, curr_grids, 10.0)

        cv2.imshow('map', d)

        cv2.waitKey(250)
        
        it = (it + 1) % 100

        # prev_frame = np.copy(curr_frame)
        prev_cells = curr_cells.copy()

def main1():

    posterior = np.zeros((500, 500))

    sense = [0] * 100
    for i in range(len(sense)):
        sense[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    prev_cells = None

    it = 0

    scan_pose = np.array([0, 0, 0])

    while 1:

        curr_frame = sense[it]

        tr = (0, 0)
        angle = 0
        scan_pose[2] = angle
        # curr_frame = do_transform(curr_frame, 5 * it, tr)

        curr_cells = slam.world_frame_to_cell_pos(curr_frame, posterior.shape,
                10.0)

        x1, y1 = slam.world_to_cell_pos(scan_pose[:2], posterior.shape,
            10.0)

        if  prev_cells is not None:

            min_sz = min(len(prev_cells), len(curr_cells))
            src = prev_cells[:min_sz]
            dst = curr_cells[:min_sz]

            theta = np.radians(angle)
            init_tr = np.array([
                [np.cos(theta), -np.sin(theta), tr[0]],
                [np.sin(theta),  np.cos(theta), tr[1]],
                [0            ,  0            , 1    ]
            ])

            transformation = icp.icp(src, dst, init_pose=init_tr)[0]

            rt = transformation[:2,:2]
            tt = transformation[:2,2].T

            curr_cells = np.dot(rt, curr_cells.T).T + tt
            curr_cells = curr_cells.round(0).astype(int)
            print(rt, tt)

        d = slam.d3_map(posterior)
        slam.draw_vertical_line(d, 250, (0, 0, 255))
        slam.draw_horizontal_line(d, 250, (0, 0, 255))

        slam.occupancy_grid_mapping(posterior,\
                np.copy(posterior), scan_pose, curr_cells, 10.0)

        cv2.imshow('map', d)

        cv2.waitKey(250)
        
        it = (it + 1) % 100

        prev_cells = np.copy(curr_cells)

def main2():

    posterior = np.zeros((500, 500))

    sense = [0] * 100
    for i in range(len(sense)):
        sense[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    prev_frame = None
    curr_frame = None

    prev_cells = None

    it = 0

    scan_pose = np.array([0, 0, 0])

    while 1:

        curr_frame = sense[it]


        tr = (0, 0)
        angle = 0
        scan_pose[2] = angle
        # curr_frame = do_transform(curr_frame, 5, tr)

        curr_grids = slam.world_frame_to_cell_pos(curr_frame, posterior.shape,
            10.0)

        x1, y1 = slam.world_to_cell_pos(scan_pose[:2], posterior.shape,
            10.0)

        curr_cells = {}
        for x0, y0 in curr_grids:
            slam.__beam_line(curr_cells, (x1, y1), (x0, y0))

        if prev_cells is not None:

            src = np.array(list(curr_cells.keys()))
            dst = np.array(list(prev_cells.keys()))

            min_sz = min(len(src), len(dst))
            src = src[:min_sz]
            dst = dst[:min_sz]

            theta = np.radians(angle)
            init_tr = np.array([
                [np.cos(theta), -np.sin(theta), tr[0]],
                [np.sin(theta),  np.cos(theta), tr[1]],
                [0            ,  0            , 1    ]
            ])

            transformation = icp.icp(src, dst, init_pose=init_tr)[0]

            rt = transformation[:2,:2]
            tt = transformation[:2,2].T

            curr_grids = np.dot(rt, curr_grids.T).T + tt
            curr_grids = curr_grids.round(0).astype(int)
            print(rt, tt)

        # d = slam.d3_map(posterior)
        # slam.draw_vertical_line(d, 250, (0, 0, 255))
        # slam.draw_horizontal_line(d, 250, (0, 0, 255))

        # occ_grids = slam.world_frame_to_cell_pos(curr_frame, posterior.shape,
        #         10.0)

        # slam.occupancy_grid_mapping(posterior,\
        #         np.copy(posterior), scan_pose, curr_grids, 10.0)

        # cv2.imshow('map', d)

        cv2.waitKey(250)
        
        it = (it + 1) % 100

        # prev_frame = np.copy(curr_frame)
        prev_cells = curr_cells.copy()

def main3():

    map_size = (500, 500)
    resolution = 10.0

    sense = [0] * 100
    for i in range(len(sense)):
        sense[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    it = 1

    prev_cells = None

    while 1:
    
        curr_cells = slam.world_frame_to_cell_pos(sense[it], map_size, resolution)

        if prev_cells is None:
            prev_cells = slam.world_frame_to_cell_pos(sense[(it - 1) % 100],\
                map_size, resolution)

        min_sz = min(len(curr_cells), len(prev_cells))
        src = curr_cells[:min_sz]
        dst = prev_cells[:min_sz]

        tr = (0, 0)
        theta = np.radians(0)
        init_tr = np.array([
            [np.cos(theta), -np.sin(theta), tr[0]],
            [np.sin(theta),  np.cos(theta), tr[1]],
            [0            ,  0            , 1    ]
        ])

        transformation = icp.icp(src, dst, init_pose=init_tr)[0]

        rt = transformation[:2,:2]
        tt = transformation[:2,2].T

        curr_cells = np.dot(rt, curr_cells.T).T + tt
        curr_cells = curr_cells.astype(int)
        print(rt, tt)

        it = (it + 1) % 100

        prev_cells = np.copy(curr_cells)

        time.sleep(0.5)

def main4():

    map_size = (500, 500)
    resolution = 10.0

    sense = [0] * 100
    for i in range(len(sense)):
        sense[i] = np.load('./data/kinect_depth2/{0}.npy'.format(i))

    it = 1

    prev_cells = None

    scan_pose = [0, 0, 0]

    while 1:

        grids = slam.world_frame_to_cell_pos(sense[it], map_size,
            10.0)
    
        x1, y1 = slam.world_to_cell_pos(scan_pose[:2], map_size,
            10.0)

        curr_cells = {}
        for x0, y0 in grids:
            slam.__beam_line(curr_cells, (x1, y1), (x0, y0))
        src = np.array(list(curr_cells.keys()))

        if prev_cells is None:
            grids = slam.world_frame_to_cell_pos(sense[(it - 1) % 100],
                map_size, resolution)
            prev_cells = {}
            for x0, y0 in grids:
                slam.__beam_line(prev_cells, (x1, y1), (x0, y0))
            dst = np.array(list(prev_cells.keys()))
        else:
            dst = prev_cells

        min_sz = min(len(src), len(dst))
        src = src[:min_sz]
        dst = dst[:min_sz]

        tr = (0, 0)
        theta = np.radians(0)
        init_tr = np.array([
            [np.cos(theta), -np.sin(theta), tr[0]],
            [np.sin(theta),  np.cos(theta), tr[1]],
            [0            ,  0            , 1    ]
        ])

        transformation = icp.icp(src, dst, init_pose=init_tr)[0]

        rt = transformation[:2,:2]
        tt = transformation[:2,2].T

        src = np.dot(rt, src.T).T + tt
        src = src.astype(int)
        print(rt, tt)

        it = (it + 1) % 100

        prev_cells = np.copy(src)

        time.sleep(0.5)

if __name__ == '__main__':
    main()
    # main1()
    # main2()
    # main3()
    # main4()
