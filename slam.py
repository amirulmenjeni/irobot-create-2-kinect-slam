
import cv2
import math
import numpy as np
import time
import bisect
import rutil
import scipy.stats as st
import random
import heapq
import logging
from data_structures import Node
from icp import icp
from sklearn.neighbors import NearestNeighbors

class ParticleFilter:

    """
    Use particle filter to implement Monte Carlo Localization (MCL).
    """

    def __init__(self, deltatime, map_size, res, num_particles=2000):

        # Compute the range of the real world coordinates that the initial grid
        # map size (map_size) holds. This is so that the particles can be
        # randomly and uniformly distributed across the map within the
        # calculated range.
        num_row, num_col = map_size
        last_row, last_col = num_row - 1, num_col - 1
        min_x, max_y = cell_to_world_pos((0, 0), map_size, res)
        max_x, min_y = cell_to_world_pos((last_row, last_col), map_size, res)
        range_x = (min_x, max_x)
        range_y = (min_y, max_y)
        range_h = (0, 360)

        self.NUM_PARTICLES = num_particles
        self.MAP_RANGE_X = range_x
        self.MAP_RANGE_Y = range_y

        # Uniform weight distribution initially.
        weight = 1 / num_particles

        self.resolution = res
        self.particles = [\
            Particle.create_random(range_x, range_y, range_h, weight) for _ in\
                range(num_particles)]
        self.deltatime = deltatime

    def update(self, z, u, x, m, occ_thres=0.75):

        """
        @param z:
            The 2-d real coordinates points projected from the depth map in the
            local reference frame.
        @param u:
            The control command for the current time step. This is a size-2
            array with values [linear_velocity, and rotational_velocity].
        @param x:
            The current state (pose) of the robot. This is a size-3 array with
            values [pos_x, pos_y, heading_radian]
        @param m:
            The occupancy grid map, a 2-D numpy array.
        @param occ_thres:
            The probability value above which a cell is considered occupied.
        """

        X = []

        z = z[::20]

        # observed_cells() use bresenham-line algorithm which is computationally
        # expensive large number of particles. Instead we can compute the local
        # observed cells once and transform later according to the pose of each
        # particle.
        end_pt_cells = world_frame_to_cell_pos(z, m.shape, self.resolution)
        origin_cell = world_to_cell_pos((0, 0), m.shape, self.resolution)
        obs_cells, cell_dict = observed_cells(origin_cell, end_pt_cells)
        obsr_mat = observation_matrix(cell_dict)

        for p in self.particles:

            i, j = world_to_cell_pos(p.x[:2], m.shape, self.resolution)

            # If a particle is in an occupied cell (i, j), set its weight to 0.
            # Otherwise, compute its new weight.
            if m[i, j] < occ_thres:
                u_noise = (1e-3, 1e-3, 3e-6, 3e-6, 3e-4, 3e-4)
                x = sample_motion_model_velocity(u, p.x, self.deltatime,\
                        u_noise)
                w = likelihood_field_measurement_model(z, p.x, obs_cells,
                        m.shape, self.resolution)
            else:
                w = 0

            p_ = Particle(x, w)

            X.append(p_)

        # Effective sample size.
        ess = len(self.particles) /(1 + ParticleFilter.cv(self.particles))

        # Resampling.
        if ess < 0.5*len(self.particles):
            self.particles = ParticleFilter.low_variance_sampler(X)

    def low_variance_sampler(particles):

        """
        Algorithm for reducing sampling error.
        """

        X = []
        M = len(particles)
        r = np.random.uniform(0, 1 / M)
        c = particles[0].w

        i = 0
        for m in range(M):

            u = r + m / M
            while u > c:
                i = i + 1
                c = c + particles[i].w

            new_x = np.copy(particles[i].x)
            new_w = 1 / M

            if particles[i].m is not None:
                new_m = np.copy(particles[i].m)
            else:
                new_m = None

            X.append(Particle(new_x, new_w, new_m))

        return X

    def select_with_replacement_resampler(particles):

        X = []
        M = len(particles)

        # Calculate the cumulative weights of the particles.
        q = np.cumsum([p.w for p in particles])

        # Generate a sorted array of random numbers between 0.0 to 1.0.
        r = np.sort(np.random.uniform(0, 1, size=(M + 1)))
        r[M] = 1.0

        i = j = 0
        index = {}
        while (i < M):
            if (r[i] < q[j]):
                index[i] = j
                i = i + 1
            else:
                j = j + 1

        for i in range(M):

            new_x = np.copy(particles[index[i]].x)
            new_w = 1 / M

            if particles[i].m is not None:
                new_m = np.copy(particles[i].m)
            else:
                new_m = None
        
            X.append(Particle(new_x, new_w, new_m))

        return X

    def cv(particles):

        """
        Returns the coefficient of variation of the particles.
        """

        M = len(particles)

        return sum([(M*p.w - 1)**2 for p in particles]) / M

class Particle:

    def __init__(self, state, weight, grid_map=None, path=None):

        self.x = state
        self.w = weight
        self.m = grid_map

    def create_random(range_x, range_y, range_h, weight):

        x = np.random.uniform(range_x[0], range_x[1])
        y = np.random.uniform(range_y[0], range_y[1])
        h = np.random.uniform(range_h[0], range_h[1])

        return Particle(np.array([x, y, h]), weight)

class FastSLAM:

    def __init__(self, map_size, resolution, dt, init_map, num_particles=150):

        self.MAP_SIZE = np.array(map_size)
        self.RESOLUTION = resolution
        self.M = num_particles
        self.DELTA_TIME = dt
        self.best_particle = 0

        self.is_map_initialized = False
        self.init_map_count = 0

        x = np.array([0, 0, 0])
        w = 1 / num_particles

        self.particles =\
                [Particle(x, w, np.copy(init_map)) for _ in range(self.M)]

    def update(self, z_t, u_t):

        end_cells = world_frame_to_cell_pos(z_t, self.MAP_SIZE, self.RESOLUTION)
        mid = np.array(self.MAP_SIZE // 2).astype(int)
        obs_dict = observed_cells(mid, end_cells)

        # An Nx3 matrix where each row is [row, col, val], and where val is the
        # update to the belief on cell (row, col) on the occupancy grid map.
        obs_mat = observation_matrix(\
            obs_dict, self.MAP_SIZE, occu=0.9, free=-0.7)

        max_weight = -1
        max_particle_index = 0

        for i, p in enumerate(self.particles):

            z_mat = np.copy(obs_mat)
            z_cells = z_mat[:,:2]

            # Prediction step: State transition of each particle.
            noise = (1e-6, 1e-6, 1e-6, 1e-6)
            p.x = sample_motion_model_odometry(u_t, p.x, noise=noise)

            H = rutil.rigid_trans_mat3(p.x)
            z_cells = rutil.transform_cells_2d(H, z_cells, self.MAP_SIZE,
                    self.RESOLUTION)

            # Correction step: Compute the weight of this particle.
            p.w = likelihood_field_measurement_model(z_cells, p.x,\
                    np.argwhere(p.m > 0.90), self.MAP_SIZE, self.RESOLUTION)

            if p.w > max_weight:
                max_weight = p.w
                max_particle_index = i

            z_mat[:,:2] = z_cells

            # Update the map of each particle.
            update_occupancy_grid_map(p.m, z_mat)

        self.best_particle = max_particle_index

        # Effective sample size.
        ess = self.M / (1 + ParticleFilter.cv(self.particles))

        if ess < 0.5 * self.M:
            self.particles = ParticleFilter.low_variance_sampler(self.particles)

    def highest_particle(self):
        return self.particles[self.best_particle]

    def estimate_pose(self):

        x_accum = 0
        y_accum = 0
        h_accum = 0
        w_accum = 0

        for p in self.particles:
            w_accum += p.w
            x_accum += p.x[0] * w_accum
            y_accum += p.x[1] * w_accum
            h_accum += p.x[2] * w_accum

        if w_accum == 0:
            return None

        x_est = x_accum / w_accum
        y_est = y_accum / w_accum
        h_est = h_accum / w_accum

        return np.array([x_est, y_est, h_est])

    def exploration(self):

        pass

def cell_to_world_pos(cell, map_size, resolution, center=False):

    """
    Returns the world position corresponding the given grid cell position.

    @param cell:
        2-tuple grid cell position (row, col), or Nx2 numpy array of grid
        cell positions.
    @param map_size:
        2-tuple shape of the map array shape (row_size, column_size).
    """

    c = np.array(cell)

    if c.ndim == 1:
        my, mx = map_size
        row, col = c
        y = (my/2 - row) * resolution
        x = (col - mx/2) * resolution
        wpos = np.array([x, y])
    else:
        sz = np.array(map_size)
        wpos = (c*np.array([-1, 1]) + sz*np.array([1/2, -1/2])) * resolution
        wpos = wpos[:,::-1]

    if center:
        wpos += np.array([resolution / 2, resolution / 2])

    return wpos

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
    Returns size-N dictionary whose keys are
    the 2-D grid map coordinate each with either value 1 or 0 indicating
    occupied or free cell.

    @param pos:
        A size-2 numpy array, representing the position of the scan device or
        robot in the the occupancy grid map.
    @param end_pt_cells:
        An Nx2 whose rows are the 2-D grid map coordinates corresponding to the
        real-world location end-point projected by the depth map.
    """

    cells = {}
    x1, y1 = pos
    for x0, y0 in end_pt_cells:
        beam_line(cells, (x1, y1), (x0, y0))

    return cells

def observation_matrix(cell_dict, map_size, occu=1, free=0):

    """
    Returns an Nx3 numpy array where each row [row, col, v] represents the row,
    column, and occupancy value of each observed cell on th grid map. 

    @param cell_dict:
        Dictionary {[row, col] => v} where [row, col] is a cell in the grid map
        and v is the occupancy value on that cell.
    """

    a = np.zeros((0, 3), dtype=np.float64)
    for i, d in enumerate(cell_dict.items()):
        row, col = d[0]
        if not is_out_of_bound((row, col), map_size):
            if d[1]:
                val = occu
            else:
                val = free
            a = np.vstack((a, [int(row), int(col), val]))
    return a

def observation_map(cell_dict, map_size):

    m = np.full(map_size, 0.5)
    for k, v in cell_dict.items():
        m[k] = v
    return m

def observation_mask(obs_map, free_thres=0.30, occu_thres=0.75):

    """
    Returns the mask corresponding to the local map obtain from the robot's
    scan.

    @param obs_map:
        An MxN array where MxN is the shape of the occupancy grid map.
    @param free_thres:
        A cell with value less than this threshold will be marked as free.
    @param occu_thres:
        A cell with value greater than this threshold will be marked as free.
    """

    assert 0.00 < free_thres < 0.50
    assert 0.50 < occu_thres < 1.00

    mask_free = cv2.inRange(obs_map, 0.00, free_thres)
    mask_occu = cv2.inRange(obs_map, occu_thres, 1.00)

    return cv2.bitwise_or(mask_free, mask_occu)

def is_out_of_bound(cell, map_size):

    """
    Checks if the cell is out of bound given the map size.
    """

    if cell[0] > (map_size[0] - 1) or cell[1] > (map_size[1] - 1) or\
            cell[0] < 0 or cell[1] < 0:
        return True      
    return False

def remove_outbound_cells(cells, map_size):

    inbound = np.argwhere(\
            (0 <= cells[:, 0]) & (cells[:, 0] < map_size[0]) &\
            (0 <= cells[:, 1]) & (cells[:, 1] < map_size[1])\
        )

    return cells[inbound]

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
            cells[(x, y)] = 0 

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
            cells[(x, y)] = 0 

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
    cells[p1] = 1

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

def occupancy_grid_mapping(posterior, prev_posterior, obs_dict):

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
    # cells = {}
    # x1, y1 = world_to_cell_pos(pose[:2], map_size, resolution)
    # for x0, y0 in scan_data:
    #     beam_line(cells, (x1, y1), (x0, y0))

    # Perform log odds update on the posterior map distribution.
    for p in obs_dict.keys():

        if is_out_of_bound(p, map_size):
            continue

        # p was observed as an obstacle.
        if obs_dict[p]:
            logit = rutil.prob_to_log_odds(prev_posterior[p]) + OCCU
            posterior[p] = rutil.log_odds_to_prob(logit)

        # p was not observed as an obstacle.
        else:
            logit = rutil.prob_to_log_odds(prev_posterior[p]) + FREE
            posterior[p] = rutil.log_odds_to_prob(logit)

def update_occupancy_grid_map(m, obs_mat):

    rows, cols = obs_mat[:,0].astype(int), obs_mat[:,1].astype(int)

    # print(rows)
    # print(cols)

    tmp = rutil.vec_prob_to_log_odds(m[rows, cols]) + obs_mat[:,2]
    m[rows, cols] = rutil.vec_log_odds_to_prob(tmp)

def update_human_grid_map(m, present_cells, present=0.9, absent=-0.5):

    prev_present_cells = np.argwhere(m > 0)

    if len(prev_present_cells) > 0:
        rows, cols = prev_present_cells.T
        tmp = rutil.vec_prob_to_log_odds(m[rows, cols]) + absent
        m[rows, cols] = rutil.vec_log_odds_to_prob(tmp)

    if len(present_cells) > 0:
        rows, cols = present_cells.T
        tmp = rutil.vec_prob_to_log_odds(m[rows, cols]) + present - absent
        m[rows, cols] = rutil.vec_log_odds_to_prob(tmp)

def human_cell_pos(m, pos, thres=0.85):

    gauss_m = np.copy(m)
    cv2.GaussianBlur(m, (3, 3), 1.5, gauss_m, 1.5)

    conds = np.logical_and(m > thres, abs(m - np.max(m)) < 1e-6)
    
    cells = np.argwhere(abs(m - np.max(m)) < 1e-6)

    if len(cells) == 0:
        return cells

    min_cell = None
    min_dist = math.inf
    for c in cells:
        d = np.linalg.norm(c - pos)
        if d < min_dist:
            min_dist = d
            min_cell = c

    return min_cell 

def cell_entropy(p):

    return -p*math.log2(p) - (1 - p)*math.log2(1 - p)

vec_cell_entropy = np.vectorize(cell_entropy)
def entropy_map(m):

    """
    @param m: The occupancy grid map.
    """

    return vec_cell_entropy(m)

def nearest_unexplored_cell(m, robot_cell, min_dist=6, unexplored_thres=0.999,\
    k=100):

    """
    @param m: The entropy map.
    """

    assert k >= 1
    assert type(k) == int
    assert min_dist >= 0
    assert 0 <= unexplored_thres <= 1
    
    X = np.argwhere(m > unexplored_thres)
    boundary = [rutil.euclidean_distance(x, robot_cell) > min_dist for x in X]
    X = X[boundary]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    dist, inds = nbrs.kneighbors([robot_cell])

    return X[inds.flatten()[np.random.randint(0, k)]]

def explore_cell(m, robot_cell, min_dist=6, k=100, tol=1e-3, entropy_thres=0.95):

    assert k >= 1
    assert type(k) == int
    assert min_dist >= 0
    assert tol > 0

    # Find unexplored cell above some entropy threshold. If entropy_thres is
    # less than or equal to 0, then find some explored cell instead.
    if entropy_thres > 0:
        X = np.argwhere((abs(m - np.max(m)) < tol) & (m > entropy_thres))
    else:
        X = np.argwhere(m < 0.5)

    boundary = [rutil.euclidean_distance(x, robot_cell) > min_dist for x in X]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    dist, inds = nbrs.kneighbors([robot_cell])

    return X[inds.flatten()[np.random.randint(0, k)]]

def local_occupancy(cell_dict, out, map_size, resolution):

    OCCU = 0.9
    FREE = -0.7

    for p in cell_dict.keys():
        
        if is_out_of_bound(p, map_size):
            continue

        if p not in out:
            out[p] = 0.5

        if cell_dict[p]:
            logit = rutil.prob_to_log_odds(out[p]) + OCCU
            out[p] = rutil.log_odds_to_prob(logit)

        else:
            logit = rutil.prob_to_log_odds(out[p]) + FREE
            out[p] = rutil.log_odds_to_prob(logit)

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
    new_pts = new_pts.round(0).astype(int)

    return new_pts 

def motion_model_velocity(succ_pose, curr_pose, control, dt,
    err_params=(1, 1, 1, 1, 1, 1)):

    """
    Closed form computation of the probability p(x_t | u_t, x_{t-1}).
    """

    assert len(err_params) == 6

    x1, y1, h1 = succ_pose 
    x0, y0, h0 = curr_pose 

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
    noise=(1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10)):

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

    return np.array([x1, y1, h1])

# def sample_motion_model_odometry(u_t, prev_x,\
#     noise=(1e-10, 1e-10, 1e-10, 1e-10)):

#     assrt len(noise) == 4

#     a1, a2, a3, a4 = noise

#     odom_pose1, odom_pose2 = u_t[0], u_t[1]

#     x1, y1, h1 = odom_pose1
#     x2, y2, h2 = odom_pose2

#     d_rot1 = math.atan2(y2 - y1, x2 - x1)
#     d_trans = math.sqrt((x2 - x1)**2, (y2 - y1)**2)
#     d_rot2 = h2 - h1 - d_rot1

#     d_rot1_sq = d_rot1**2
#     d_trans_sq = d_trans**2
#     d_rot2_sq = d_rot2**2

#     d_rot1_ = d_rot1 - sample_normal_distribution(a1*d_rot1_sq + a2*d_trans_sq)
#     d_trans_ = d_trans - sample_normal_distribution(\
#         a3*d_trans_sq + a4*d_rot1_sq + a4*d_rot2_sq)
#     d_rot2_ = d_rot2 - sample_normal_distribution(a1*d_rot2_sq + a2*d_trans_sq)

#     return

def sample_motion_model_odometry(u_t, prev_x,\
    noise=(1e-10, 1e-10, 1e-10, 1e-10)):

    assert len(noise) == 4

    a1, a2, a3, a4 = noise

    d_dist, d_rad = u_t

    d_dist_sq = d_dist**2
    d_rad_sq = d_rad**2

    d_dist_ = d_dist - sample_normal_distribution(a1*d_rad_sq + a2*d_dist_sq)
    d_rad_ = d_rad - sample_normal_distribution(a3*d_dist_sq + a4*d_rad_sq)

    x, y, h = prev_x
    x = x + d_dist_ * math.cos(h + d_rad_)
    y = y + d_dist_ * math.sin(h + d_rad_)
    h = h + d_rad_

    return np.array([x, y, h])

def likelihood_field_measurement_model(end_cells, x, occ_cells, map_size, res):

    """
    Compute the probability p(z_t, | x_t, m).

    @param z:
        An Nx2 array for N observed obstacle points in real 2-D coordinates.
    @param x:
        The state or pose of the particle.
    @param occ_cells:
        An Mx2 array for M obstacle cells in the occupancy grid map.
    """

    if len(occ_cells) == 0:
        return 0

    H = rutil.rigid_trans_mat3(x)
    end_cells = rutil.transform_cells_2d(H, end_cells, map_size, res)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(end_cells)

    dist, inds = nbrs.kneighbors(occ_cells)

    ave_min_dist = 0
    for d in dist[0]:
        ave_min_dist += d

    return np.exp(-ave_min_dist / len(dist))

def correlation_measurement_model(obsr_mat, grid_map, resolution):

    """
    Compute the probability p(z_t | x_t, m) i.e., the probability that the
    current observation z_t is true given our knowledge of the state x_t and the
    map m.

    Here, x_t corresponds to the pose, m corresponds to the ogm_posterior, and
    z_t corresponds to the scan_end_pts.

    Ref: Probabilistic Robotics, Ch. 4.
         See also https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """

    n = obsr_mat.shape[0]

    glob_mat = np.zeros((n, 3))
    for i, k in enumerate(obsr_mat):
        row, col, val = k
        row, col = int(row), int(col)
        glob_mat[i] = [row, col, grid_map[row, col]]

    m = np.sum(obsr_mat[:,2] + glob_mat[:,2]) / (2*n)
    a = np.sum((glob_mat[:,2] - m) * (obsr_mat[:,2] - m))
    b = np.sum((glob_mat[:,2] - m)**2)
    c = np.sum((obsr_mat[:,2] - m)**2)
    pcc = a / (math.sqrt(b*c))

    return max(0, pcc)

def correlation_measurement_model2(obs_map, pose, grid_map, mask, resolution):

    # Convert the spatial measurement to grid map's resolution. Used for
    # applying transformation on 2-D array map.
    grid_pose = np.array([pose[0] / resolution, pose[1] / resolution, pose[2]])

    # Get the mask that cover the observation map on the global map. Transform
    # this mask to the orient it in the global reference frame.
    mask = rutil.transform_map(grid_pose, mask)

    # Apply the mask to the grid map, and transform it back to the robot's local
    # reference frame so we can compare the masked grid map and the local map.
    gmap = cv2.bitwise_and(grid_map, grid_map, mask=mask)
    gmap = rutil.transform_map(-grid_pose, gmap)

    gmap = gmap[gmap > 1e-3]
    lmap = obs_map[abs(obs_map - 0.5) > 1e-6]

    # # The number of overlapping cells size of lmap or size of gmap.
    n = len(lmap)

    # # The mean distribution of global and local observation map.
    m = np.sum(gmap + lmap) / (2*n)

    # # Calculate Pearson correlation coefficient (PPC) between both map.
    a = np.sum((gmap - m) * (lmap - m))
    b = np.sqrt(np.sum((gmap - m)**2) * np.sum((lmap - m)**2))
    ppc = a/b

    # Since PPC can have values from -1 to +1, we take the negative values as 0.
    # Return the probability p(z_t, | x_t, m).
    return max(0, ppc)

def sample_normal_distribution(b_sq):

    """
    Normal distribution with zero-centered mean.
    """

    return np.random.normal(0, math.sqrt(b_sq))

def prob_normal_distribution(a, b_sq):

    """
    Finds the probability of argument a under a zero-centered distribution with
    variance b_sq.
    """

    return math.exp(-(a**2) /  (2*b_sq)) / math.sqrt(2*math.pi*b_sq)

def posterior_prob_dist(posterior):

    """
    Convert the posterior probabilility of each cell in the posterior from log
    odds form to real probability with values ranging from 0.0 to 1.0.
    """

    log_odds_vector = np.vectorize(rutil.log_odds_to_prob)
    return log_odds_vector(posterior)

def d3_map(posterior, invert=False):

    """
    Returns the posterior in log odds form into an image with three channels
    (BGR). The probability value in each cell is normalized from 0 to 255.
    """

    d = (posterior * 255).astype(np.uint8)
    d = np.dstack((d, d, d))

    if invert:
        d = 255 - d

    return d

def random_explore_cell(pose, grid_map, free_thres, min_radius, max_radius, res):

    robot_cell = world_to_cell_pos(pose[:2], grid_map.shape, res)
    free_cells = np.argwhere(grid_map < free_thres)

    # Manhattan distance.
    distance = np.sum(np.abs(free_cells - np.array(robot_cell)), axis=1)

    lower_bound = min_radius <= distance
    upper_bound = distance <= max_radius

    free_cells = free_cells[np.logical_and(lower_bound, upper_bound)]

    return random.choice(free_cells)

def neighbor_cells(cell, grid_map, occu_thres, kernel_radius):

    row, col = cell
    for i in range(-1, 2):
        for j in range(-1, 2):

            if i == 0 and j == 0:
                continue

            neighbor = (row + i, col + j)

            if not is_out_of_bound(neighbor, grid_map.shape) and\
               not is_colliding(neighbor, grid_map, occu_thres, kernel_radius):

                yield neighbor

def is_colliding(cell, grid_map, occu_thres, kernel_radius):

    row, col = cell

    for i in range(row - kernel_radius, row + kernel_radius + 1):
        for j in range(col - kernel_radius, col + kernel_radius + 1):

            if grid_map[i, j] >= occu_thres:
                return True
    return False

def path_cost(cell, grid_map, occu_thres, kernel_radius):

    row, col = cell
    count = 1

    for i in range(row - kernel_radius, row + kernel_radius + 1):
        for j in range(col - kernel_radius, col + kernel_radius + 1):

            if grid_map[i, j] >= occu_thres:
                count += 1

    return count / ((kernel_radius + 1) ** 2)

def __heapsort(iterable):
    
    h = []
    for v in iterable:
        heapq.heappush(h, v)
    return [heapq.heappop(h) for _ in range(len(h))]

def __replace_greater(queue, node):

    for i in range(len(queue)):
        if queue[i].label == node.label and queue[i].f_cost() > node.f_cost():
            queue[i] = node

    return __heapsort(queue)

def shortest_path(start, goal, grid_map, occu_thres, kernel_radius=1):

    """
    A* algorithm to find shortest-path from the starting cell to the goal cell
    on the given grid map.

    @param start:
        The starting cell.
    @param goal:
        The goal cell.
    @param grid_map:
        The occupancy grid map representing the occupancy probability of each
        cell on the map.
    @occ_thres:
        The threshold occupancy value, the minimum probability value of a cell
        to be labeled as occupied.
    """

    goal = tuple(goal)
    kernel_radius = np.array(kernel_radius)

    assert grid_map[goal[0], goal[1]] < occu_thres
    assert kernel_radius > 0

    kernel_radius = int(kernel_radius)

    # Heuristic function.
    # h = lambda n : np.sum(np.abs(np.array(n.label) - np.array(goal)))
    h = lambda n: rutil.euclidean_distance(n.label, goal)
    
    # Path-cost function.
    g = lambda n : n.parent.g_cost +\
        path_cost(n.label, grid_map, occu_thres, kernel_radius)

    explored_set = {}
    frontier_set = {}

    node = Node(start)
    node.g_cost = 0
    node.h_cost = h(node)
    frontiers = []
    heapq.heappush(frontiers, node)
    frontier_set[node.label] = True

    solution = []

    while 1:

        if len(frontiers) == 0:
            return []

        node = heapq.heappop(frontiers)
        if node.label in frontier_set:
            del frontier_set[node.label]

        # Goal-test.
        if node.label == goal:
            while node.parent is not None:
                solution.append(node.label)
                node = node.parent
            return solution[::-1]

        explored_set[node.label] = True

        for neighbor in neighbor_cells(node.label, grid_map, occu_thres,\
                kernel_radius):

            child_node = Node(neighbor)
            child_node.parent = node
            child_node.g_cost = g(child_node)
            child_node.h_cost = h(child_node)

            if neighbor not in explored_set and\
               neighbor not in frontier_set:

                heapq.heappush(frontiers, child_node)
                frontier_set[neighbor] = True

            elif neighbor in frontier_set:
                frontiers = __replace_greater(frontiers, child_node)
