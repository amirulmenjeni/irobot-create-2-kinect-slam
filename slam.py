
import cv2
import math
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

def transform_2d_points(a, m):

    """
    @param a:
        A Nx2 numpy array containing a list of 2d points.
    @param m:
        A homogeneous transformation matrix.
    """

    a = np.hstack( (a, np.ones((a.shape[0], 1))) )

    for i in range(a.shape[0]):
        a[i] = m @ a[i]

    return a[:, :2]

def log_odds_to_prob(k):
    return 1.0 - (1.0 / (1.0 + math.exp(k)))

def prob_to_log_odds(p):
    return math.log10(p / (1.0 - p))

def cell_to_world_pos(cell, map_size, resolution):

    """
    Returns the world position corresponding the given grid cell position.

    @param cell:
        2-tuple grid cell position (column, row).
    @param map_size:
        2-tuple shape of the map array shape (row_size, column_size).
    """

    mx, my = map_size
    cx, cy = cell
    x = (cx - mx / 2) * resolution
    y = (my / 2 - cy) * resolution
    return (x, y)

def world_to_cell_pos(wpos, map_size, resolution):

    mx, my = map_size
    wx, wy = wpos
    x = (wx / resolution) + (mx / 2)
    y = (my / 2) - (wy / resolution)

    return (int(y), int(x)) # row, column

def world_frame_to_cell_pos(frame, map_size, resolution):

    """
    @param frame:
        An Nx2 array of the 2d real coordinates.
    @param map_size:
        The occupancy grid map size, a tuple of (num_row, num_col).
    @param resolution:
        The units per pixel of the grid map.
    """

    sz = np.array(map_size) / 2
    cell_pos = (frame / resolution) * np.array([1, -1]) + sz

    return cell_pos[:,::-1].astype(int) # to row, column

def __is_out_of_bound(cells, map_size):

    if cells[0] >= (map_size[1] - 1) or cells[1] >= map_size[0] or\
       cells[0] < 0 or cells[1] < 0:
           return True
    return False

def __line_low(cells, p0, p1):

    x0, y0 = p0
    x1, y1 = p1

    dx = x1 - x0
    dy = y1 - y0

    yi = 1

    if dy < 0:
        yi = -1
        dy = -dy

    d = 2 * dy - dx
    y = y0

    for x in range(x0, x1 + 1):

        if (x, y) not in cells:
            cells[(x, y)] = False

        if d > 0:
            y = y + yi
            d = d - 2 * dx
        d = d + 2 * dy

def __line_high(cells, p0, p1):

    x0, y0 = p0
    x1, y1 = p1

    dx = x1 - x0
    dy = y1 - y0

    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx

    d = 2 * dx - dy
    x = x0

    for y in range(y0, y1 + 1):

        if (x, y) not in cells:
            cells[(x, y)] = False

        if d > 0:
            x = x + xi
            d = d - 2 * dy
        d = d + 2 * dx

def beam_line(cells,  p0, p1):

    """
    Returns the cells covered by the beam from point p0 to p1.
    Ref: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    @param p0: The cell where the robot is located.
    @param p1: The cell where the obstacle is detected.
    """

    x0, y0 = p0
    x1, y1 = p1

    # Mark the cell as occupied. This is does not reflect the distribution of
    # the map. This simply mark p1 as the cell where an obstacle is detected.
    cells[p1] = True

    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            __line_low(cells, p1, p0)
        else:
            __line_low(cells, p0, p1)
    else:
        if y0 > y1:
            __line_high(cells, p1, p0)
        else:
            __line_high(cells, p0, p1)

def occupancy_grid_mapping(posterior, prev_posterior, pose, scan_data,
    resolution):

    map_size = posterior.shape

    OCCU = 0.9
    FREE = -0.7

    # For each obstacle position (b_i) derived from the depth sensor data, use
    # Bresenham's line algorithm to get the set of cells C_i that intersects
    # with a straight line formed between the robot and the given obstacle b_i
    # position.  Take the union C = C_0 U C_1 U ... U C_N.
    cells = {}
    x1, y1 = world_to_cell_pos(pose[:2], map_size, resolution)
    for x0, y0 in scan_data:
        beam_line(cells, (x1, y1), (x0, y0))

    # Perform log odds update on the posterior map distribution.
    for p in cells.keys():

        if __is_out_of_bound(p, map_size):
            continue

        # p was observed as an obstacle.
        if cells[p]:
            posterior[p] = min(prev_posterior[p] + OCCU, 100)

        # p was not observed as an obstacle.
        else:
            posterior[p] = min(prev_posterior[p] + FREE, 100)


def d3_map(posterior):

    log_odds_vector = np.vectorize(log_odds_to_prob)
    d = (log_odds_vector(posterior) * 255).astype(np.uint8)
    d = np.dstack((d, d, d))

    return d

def icp(prev, curr, init_tr=None, iterations=50, sd_mult=1.0, out=None):

    """
    @param prev: A m-by-2 numpy array where m is the number of 2-d points from
    previous scan.
    @param curr: A m-by-2 numpy array where m is the number of 2-d points from
    current scan.
    @param init_tr: Initial transformation of the robot in [dx, dy, d_radian]^T
    @iterations: The number of iterations for the ICP algorithm.

    @output r, t: Returns a 2-by-2 rotation matrix and a size-2 array
    translation vector to transform the points in prev to match the points in
    curr.
    """

    # Refs: https://stackoverflow.com/questions/20120384/,
    #       https://nghiaho.com/?page_id=671

    # Make src and dst having the same size. Usually the difference of data size
    # from each scan is very small. The difference is the result from ignoring
    # max length reading during each scan.
    min_sz = min(prev.shape[0], curr.shape[0])
    src = np.copy(prev[:min_sz])
    dst = np.copy(curr[:min_sz])

    # Initinialize the transformation with initial pose estimation.
    if init_tr is not None:
        x0, y0 = init_tr[:2]
        h0 = init_tr[2]

        tr = np.array([
            [math.cos(h0), -math.sin(h0), x0],
            [math.sin(h0),  math.cos(h0), y0],
            [0           ,  0           , 1 ]
        ])
    else:
        tr = np.identity(3)

    # Apply the rotation then translation.
    src = np.dot(tr[:2,:2], src.T).T + tr[:2,2]

    min_mse = math.inf

    for _ in range(iterations):

        # Use Neares Neighbors algorithm to find the closest point in dst for
        # each point in src.
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)

        distances, indices = nbrs.kneighbors(src)
        distances = distances.flatten()
        indices = indices.flatten()
        dst = dst[indices]

        # Use Singular Valued Decomposition (SVD):
        # U, S, Vt = SVD(A)
        #
        # The rotation R can be computed as:
        # R = Vt @ transpose(U)
        #
        # And the translation:
        #
        # T = -R @ centroid_src + centroid_dst

        # mn = np.mean(distances)
        # sd = np.std(distances)

        # inds_pair = np.where(abs(distances - mn) <= sd * sd_mult)[0]
        # psrc = src[inds_pair]
        # pdst = dst[inds_pair]

        centroid_src = np.mean(src, axis=0)
        centroid_dst = np.mean(dst, axis=0)

        centered_src = src - centroid_src
        centered_dst = dst - centroid_dst

        s = centered_src.T @ centered_dst
        u, _, v = np.linalg.svd(s)

        rt = v @ u.T

        # Rare case: reflection.
        if np.linalg.det(rt) < 0:
            rt[:,1] = -rt[:,1]

        tt = centroid_dst.reshape(1, 2) - centroid_src.reshape(1, 2) @ rt

        mse = ((dst - src) ** 2).mean()
        if mse < min_mse:
            min_mse = mse
        else:
            print('mse:', mse)
            break

        # Apply the transformation on src for this iteration, moving the points
        # in src closer to dst.
        src = np.dot(rt, src.T).T + tt

        tmp = np.hstack((rt, tt.reshape(2, 1)))
        tmp = np.vstack((tmp, [0.0, 0.0, 1.0]))

        tr = tr @ tmp

    return tr[:2, :2], tr[:2, 2].reshape(2), src

def draw_square(d3_map, resolution, pose, bgr, width=3):

    map_size = d3_map.shape[:2]
    y0, x0 = world_to_cell_pos(pose[:2], map_size, resolution)

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

def draw_vertical_line(d3_map, x_pos, bgr):

    for k in range(3):
        d3_map[:, x_pos, k] = bgr[k]

def draw_horizontal_line(d3_map, y_pos, bgr):

    for k in range(3):
        d3_map[y_pos, :, k] = bgr[k]
