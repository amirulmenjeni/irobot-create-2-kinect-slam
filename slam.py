
import cv2
import math
import numpy as np
import time
import rutil
from icp import icp
from sklearn.neighbors import NearestNeighbors

class ParticleFilter():

    def __init__(self, num_particles=2000):
        pass

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

    """
    Convert the log odds notation k to the corresponding probability value
    ranging [0, 1].
    """

    return 1.0 - (1.0 / (1.0 + math.exp(k)))

def prob_to_log_odds(p):

    """
    Convert the probability value p to its log odds notation.
    """

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

    """
    Convert the given real 2-D point position in the global reference frame to
    the corresponding 2-D grid coordinate position (row, column).

    @param wpos:
        Real 2-D position in the global reference frame.
    @param map_size:
        A 2-tuple representing the number of rows and the number of column of
        the grid map.
    @param resolution:
        The resolution of the gird map, or the units per pixel of the grid map.
        The higher the resolution, the coarser the grid map.
    """

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

def observed_cells(pos, end_pt_cells):

    """
    Returns an Nx2 numpy array whose rows are the 2-D grid map coordinates of
    all cells observed in the current scan.

    @param pos:
        A size-2 numpy array, representing the position of the scan device or
        robot in the global reference frame of the occupancy grid map.
    @param end_pt_cells:
        An Nx2 whose rows are the 2-D grid map coordinates corresponding to the
        real-world location end-point projected by the depth map.
    """

    cells = {}
    x1, y1 = pos
    for x0, y0 in end_pt_cells:
        beam_line(cells, (x1, y1), (x0, y0))

    return np.array(list(cells.keys())), cells

def __is_out_of_bound(cells, map_size):

    """
    Checks if the cell is out of bound given the map size.
    """

    if cells[0] >= (map_size[1] - 1) or cells[1] >= map_size[0] or\
       cells[0] < 0 or cells[1] < 0:
           return True
    return False

def __line_low(cells, p0, p1):

    """
    Function implementation for beam_line()
    """

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

    """
    Function implementation for beam_line()
    """

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

    """
    Calculates the posterior distribution m over the map given the observation
    data z and odometry pose data x: 

    p(m | z_{1:t}, x_{1:t})

    @param posterior:
        The posterior distribution of the grid map in log odds form that is to
        be updated.
    @param prev_posterior:
        The current posterior distribution, same format as posterior. Generally
        this can be a copy of the posterior array.
    @param pose:
        The current pose of the robot.
    @param scan_data:
        The 2-D grid map coordinate (row, column) projected from the depth
        scan data.
    @param resolution:
        The resolution of the grid map. A value of 1 means 1 pixel represent 1
        unit of metric spatial measurement.
    """

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

def scan_match(src_pts, dst_pts, pts):

    """
    Find the rigid transformation H that reduces the mean squared error between
    the point sets src_pts and dst_pts, and return new points after applying H
    to pts. The two point sets should have many overlapping points to increase
    the accuracy of H.

    Note that the row size of src_pts and dst_pts should be more or less the
    same size (i.e. K ~= J). D is the dimension of the point sets.

    @param src_pts:
        An KxD numpy array where each row is size-D array.
    @param dst_pts:
        An JxD numpy array where each row is size-D array.
    @param pts:
        An LxD numpy array where each row is size-D array.
    """

    min_sz = min(src_pts.shape[0], dst_pts.shape[0])
    src = src_pts[:min_sz]
    dst = dst_pts[:min_sz]
    
    tran, _, _ = icp.icp(src, dst, max_iterations=30)

    Rt = tran[:2,:2]
    Tt = tran[:2,2].T

    new_pts = np.dot(Rt, pts.T).T + Tt
    new_pts = matched_cells.round(0).astype(int)

    return new_pts 

def motion_model_velocity(succ_pose, curr_pose, control, dt,
    err_params=(1, 1, 1, 1, 1, 1)):

    """
    Closed form computation of the probability p(x_t | u_t, x_{t-1}).
    """

    assert len(err_params) == 6

    x1, y1, h1 = succ_pose 
    x0, y0, h0 = curr_pose 

    h1 = math.radians(h1)
    h0 = math.radians(h0)

    dx = x0 - x1
    dy = y0 - y1

    mu = 0.5 * ( (dx*math.cos(h0) + dy*math.sin(h0)) /\
                (dy*math.cos(h0) - dx*math.cos(h0)) )

    x_ = ((x0 + x1) / 2.0) + mu*dy
    y_ = ((y0 + y1) / 2.0) + mu*(-dx)

    r_ = math.sqrt((x0 - x_)**2 +(y0 - y_)**2)

    da = math.atan2(y1 - y_, x1 - x_) - math.atan2(y0 - y_, x0 - x_)

    v_ = (da / dt) * r_
    w_ = da / dt

    gamma = ((h1 - h0) / dt) + w_

    v, w = control

    a1, a2, a3, a4, a5, a6 = err_params

    v_sq = v**2
    w_sq = w**2
    
    p1 = np.random.normal(v - v_, a1*v_sq + a2*w_sq)
    p2 = np.random.normal(w - w_, a3*v_sq + a4*w_sq)
    p3 = np.random.normal(gamma, a5*v_sq + a6*w_sq)

    return p1*p2*p3

def sample_motion_model_velocity(control, curr_pose, dt,
    noise=(1, 1, 1, 1, 1, 1)):

    """
    Sample x_t from the probability p(x_t | u_t, x_{t-1}). This can be used to
    compute the probability of a particle in particle filter.

    @param control:
        The control issued in the current timestep (i.e., forward velocity and
        rotational velocity), a size-2 numpy array.
    @param curr_pose:
        The pose of the robot in the current timestep.
    @param dt:
        The fixed time period between timesteps.
    @noise:
        The noise parameters affecting the motion. The first and the second
        parameters adjust the noise of forward velocity. The third and fourth
        parameters adjust the noise of rotational velocity. The fifth and sixth
        parameters adjust the noise perturbing the motion in general.
    """

    assert len(noise) == 6

    a1, a2, a3, a4, a5, a6 = noise

    x, y, h = curr_pose
    h = math.radians(h)

    v, w = control

    v_sq = v**2
    w_sq = w**2

    v_ = v + sample_normal_distribution(a1*v_sq + a2*w_sq)
    w_ = w + sample_normal_distribution(a3*v_sq + a4*w_sq)
    gamma = sample_normal_distribution(a5*v_sq + a6*w_sq)

    r_ = (v_ / w_)

    x1 = x - r_*math.sin(h) + r_*math.sin(h + w_*dt)
    y1 = y + r_*math.cos(h) - r_*math.cos(h + w_*dt)
    h1 = h + w_*dt + gamma*dt

    return np.array([x1, y1, np.degrees(h1)])

def likelihood_field_model(scan_data, curr_pose, posterior):

    """
    Compute the probability p(z[k]_t, | x_t, m).
    """

    q = 1

    # Weight of probability for the depth scan z to hit the "True" range,
    # weight of failure (max value measurement) probability, and weight of
    # random noise probability. 
    z_hit = 1
    z_max = 1
    z_random = 1

    # The standard deviation of the 1-D Gaussian distribution of measurement
    # noise of z.
    sigma_hit = 1

    grid_map = posterior_prob_dist(posterior_map)

    nbrs = NearestNeighbors(n_neighbors=1).fit(\
        np.argwhere(grid_map > 0.5))

    for row, col in scan_data:

        if grid_map[row, col] == 0.5:
            k = 1 / z_max
        else:
            src = np.array(row, col).reshape(-1, 1)

            # Get the nearest obstacle cell.
            dist, inds = nbrs.kneighbors(src)

            k = prob_normal_distribution(dist, sigma_hit**2) + (z_random / z_max)

        q = q * k

    return q

def map_matching_measurement_model(pose, scan_end_pts, ogm_posterior, ogm_res):

    """
    Compute the probability p(z_t | x_t, m) i.e., the probability that the
    current observation z_t is true given our knowledge of the state x_t and the
    map m.

    Here, x_t corresponds to the pose, m corresponds to the ogm_posterior, and
    z_t corresponds to the scan_end_pts.

    @param pose:
        Pose of robot or particle, a size-3 1-D numpy array.
    @param scan_end_pts:
        An Nx3 numpy array consisting of the real 2-D coordinates projected by
        the depth scan map.
    @param ogm_posterior:
        The posterior distribution in log odds form of the occupancy grid map.
    @param ogm_res:
        The resolution of the occupancy grid map ogm_posterior.
    """

    rpos_grid = world_to_cell_pos(pose[:2], ogm_posterior.shape, ogm_res)

    grid_map = posterior_prob_dist(ogm_posterior)

    # Transform the scan_end_pts to the global reference frame, so that they
    # share the same reference frame as the occupancy grid map to be compared.
    H = rutil.rigid_trans_mat3(pose)
    scan_end_pts = rutil.transform_pts_2d(H, scan_end_pts)

    # The cell locations of the local map generated by the scan z_t.
    end_pt_cells = world_frame_to_cell_pos(scan_end_pts, ogm_posterior.shape,
        ogm_res)
    loc_cells, local_map = observed_cells(rpos_grid, end_pt_cells)

    # The number of overlaps between local map from recent scan and the global
    # map (occupancy grid map).
    n = len(loc_cells)

    locs = loc_cells.tolist()

    # Calculate the mean map.
    m = sum([grid_map[i,j] + local_map[i,j] for (i,j) in locs]) / (2*n)

    # Calculate the correlaction between the local map (generatd from the scan
    # z_t) and the global map. The correlation scales between -1 to +1. Map
    # matching interprets the value max(a/b, 0) as the probability p(m_local |
    # x_t, m) which substitutes the probability p(z_t | x_t, m) when the local
    # map is generated from the scan z_t.
    a = sum([(grid_map[i,j] - m) * (local_map[i,j] - m) for (i,j) in locs])
    b = sum([(grid_map[i,j] - m)**2 for (i,j) in locs]) *\
        sum([(local_map[i,j] - m)**2 for (i,j) in locs])
    b = math.sqrt(b)
    prob = max(a/b, 0)

    return prob

def sample_normal_distribution(b_sq):

    """
    Normal distribution with zero-centered mean.
    """

    return np.random.normal(0, math.sqrt(b_sq))

def prob_normal_distribution(a, b_sq):

    return math.sqrt(2*math.pi*b_sq) * math.exp(-0.5*(a**2) / b_sq)

def posterior_prob_dist(posterior):

    """
    Convert the posterior probabilility of each cell in the posterior from log
    odds form to real probability with values ranging from 0.0 to 1.0.
    """

    log_odds_vector = np.vectorize(log_odds_to_prob)
    return log_odds_vector(posterior)

def monte_carlo_localization(particles):

    pass

def d3_map(posterior, invert=False):

    """
    Returns the posterior in log odds form into an image with three channels
    (BGR). The probability value in each cell is normalized from 0 to 255.
    """

    posterior_distribution = posterior_prob_dist(posterior)
    d = (posterior_distribution * 255).astype(np.uint8)
    d = np.dstack((d, d, d))

    if invert:
        d = 255 - d

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
