
import frame_convert2
import cv2
import math
import numpy as np
import time
import freenect
import ICP
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
    return np.array([ pose[0], pose[1], p[0], p[1] ])

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

def occupancy_grid_mapping(posterior, prev_posterior, pose, obs_data,
    resolution):

    map_size = posterior.shape

    OCCU = 0.9
    FREE = -0.7

    # For each obstacle position (b_i) derived from the depth sensor data, use
    # Bresenham's line algorithm to get the set of cells C_i that intersects
    # with a straight line formed between the robot and the given obstacle b_i
    # position.  Take the union C = C_0 U C_1 U ... U C_N.
    cells = {}
    for x0, y0, x1, y1 in obs_data:

        x0, y0 = world_to_cell_pos((x0, y0), map_size, resolution)

        x1, y1 = world_to_cell_pos((x1, y1), map_size, resolution)
        __beam_line(cells, (x0, y0), (x1, y1))

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

def icp(prev, curr, init_pose=(0, 0, 0), iterations=5):

    # Ref: https://stackoverflow.com/questions/20120384/

    x0, y0 = init_pose[:2]
    h0 = init_pose[2]

    src = np.copy(prev)
    dst = np.copy(curr)

    # Initinialize the transformation with initial pose estimation.
    tr = np.array([
        [math.cos(h0), -math.sin(h0), x0],
        [math.sin(h0),  math.cos(h0), y0],
        [0           ,  0           , 1 ]
    ])

    src = transform_2d_points(src, tr).astype(int)

    for i in range(iterations):

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
    
        _, indices = nbrs.kneighbors(src)

        t = cv2.estimateAffinePartial2D(src, dst)[0]
        t = np.vstack((t, [0, 0, 1]))

        src = transform_2d_points(src, t).astype(int)

        tr = tr @ t

    return tr

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
