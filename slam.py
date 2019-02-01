
import frame_convert2
import cv2
import math
import numpy as np
import time
import freenect
import vtk
from sklearn.neighbors import NearestNeighbors

def transform_2d_points(a, m):

    """
    @param a: A Nx2 numpy array containing a list of 2d points.
    @param m: A homogeneous transformation matrix.
    """

    a = np.hstack( (a, np.ones((a.shape[0], 1))) )

    for i in range(a.shape[0]):
        a[i] = m @ a[i]

    return a[:, :2]

def log_odds_to_prob(k):
    return 1.0 - (1.0 / (1.0 + math.exp(k)))

def prob_to_log_odds(p):
    return math.log10(p / (1.0 - p))

def depth_to_world_2d_pos(x_scr, y_scr, depth, scr_width, hfov, pose):

    """
    The Kinect depth sensor has depth value for each "pixel". This function
    returns the real world position (x1, y1) of the obstacles detected by the
    depth sensor when the robot is assumed to be at position (x0, y0).

    @param x_scr: The column corresponding the depth value.
    @param y_scr: The row corresponding the depth value.
    @param depth: The depth value at (x_scr, y_scr).
    @param scr_width: The width (number of columns in a row) that the depth data
    has. Xbox 360 Kinect has 480 columns in a row (480x640) depth data
    resolution. 
    @param hfov: The horizontal field of view of the depth sensor.
    Xbox 360 Kinect has hfov ~ 58.5.
    """

    hfov = math.radians(hfov)
    heading = math.radians(pose[2])

    # Focal length.
    f = scr_width / (2.0 * math.tan(hfov / 2.0))

    x_world = depth * x_scr / f
    y_world = depth * y_scr / f

    # Fix original heading along the x-axis.
    theta = (hfov / 2.0) - (math.pi / 2.0) + heading

    r = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0                ,  0                , 1]
    ])

    p = np.array([x_world, y_world, 1])
    p = np.matmul(r, p)
    return p[:2]

def cell_to_world_pos(cell, map_size, resolution):

    """
    Returns the world position corresponding the given grid cell position.

    @param cell: 2-tuple grid cell position (column, row).
    @param map_size: 2-tuple shape of the map array shape 
    (row_size, column_size).
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
    return (int(y), int(x))

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

def __beam_line(cells,  p0, p1):

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
        __beam_line(cells, (x1, y1), (x0, y0))

    # Perform log odds update on the posterior map distribution.
    for p in cells.keys():

        if __is_out_of_bound(p, map_size):
            print('!')
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

def icp(prev, curr, init_tr=(0, 0, 0), iterations=50):

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

    x0, y0 = init_tr[:2]
    h0 = init_tr[2]

    src = np.copy(prev)
    dst = np.copy(curr)

    # Initinialize the transformation with initial pose estimation.
    tr = np.array([
        [math.cos(h0), -math.sin(h0), x0],
        [math.sin(h0),  math.cos(h0), y0],
        [0           ,  0           , 1 ]
    ])

    # Calculate the geometrical centroid of each point cloud.
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)

    # Apply the rotation then translation.
    src = np.dot(tr[:2,:2], src.T).T + tr[:2,2]

    initial_mean_dist = None

    for _ in range(iterations):

        # Use Neares Neighbors algorithm to find the closest point in dst for
        # each point in src.
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
    
        distances, indices = nbrs.kneighbors(src)

        if initial_mean_dist is None:
            initial_mean_dist = np.mean(distances)

        # The ratio or percentage of how close src point cloud is to the dst
        # point cloud. A ratio of 0 means src is perfectly matched with dst
        # (each point in src has zero distance to the corresponding cloesest
        # point in dst). A ratio of 1 means src and dst is still in the initial
        # state. This ratio should decrease every iteration.
        ratio = np.mean(distances) / initial_mean_dist

        # Prevent further unnecessary iterations. When the src is 95% closer to
        # dst from its initial state, we stop.
        if ratio < 0.05:
            break

        # Use Singular Valued Decomposition (SVD):
        # U, S, Vt = SVD(A)
        #
        # The rotation R can be computed as:
        # R = Vt @ transpose(U)
        #
        # And the translation:
        #
        # T = -R @ centroid_src + centroid_dst

        a = np.zeros((2, 2))
        for p_src, p_dst in zip(src, dst[indices.flatten()]):
            p_src = p_src - centroid_src
            p_dst = p_dst - centroid_dst
            a += p_src.reshape(2, 1) @ p_dst.reshape(1, 2)
        u, s, v = np.linalg.svd(a)

        r = v @ np.transpose(u)
        t = -r @ centroid_src.reshape(2, 1) + centroid_dst.reshape(2, 1)

        # Apply the transformation on src for this iteration, moving the points
        # in src closer to dst.
        src = np.dot(r, src.T).T + t.reshape(2)
        src = src.astype(int)

        # Recalculate the geometrical centroid for the new src point cloud.
        centroid_src = np.mean(src, axis=0)

        tmp = np.hstack((r, t))
        tmp = np.vstack((tmp, [0, 0, 1]))

        tr = tr @ tmp

    return tr[:2, :2], tr[:2, 2].reshape(2)

def draw_square(d3_map, resolution, pose, bgr, r=3):

    map_size = d3_map.shape[:2]
    y0, x0 = world_to_cell_pos(pose[:2], map_size, resolution)

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
