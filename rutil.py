import math
import numpy as np
import datetime
import cv2
import socket
import os
import tables
from scipy import ndimage

def rigid_trans_mat3(pose):

    """
    Returns a rigid homogeneous transformation matrix.

    @param pose:
        A size-3 1-D numpy array [x, y, heading]^T where the heading is in
        radians.
    """

    x, y, r = pose

    return np.array([
            [math.cos(r), -math.sin(r), x],
            [math.sin(r),  math.cos(r), y],
            [0          ,  0          , 1]
        ])

def transform_pts_2d(H, pts):

    """
    Return new points after transforming all the points in pts by h.

    @param H:
        A 3x3 homogenous transformation matrix.
    @param pts:
        An Nx2 numpy array. Each row in the array is a size-2 array representing
        real 2-dimensional point.
    """

    assert H.shape[0] == H.shape[1] and H.shape[0] == 3
    assert pts.shape[1] == 2

    return np.dot(H[:2,:2], pts.T).T + H[:2,2]

def transform_cells_2d(H, cells, map_size, res):

    mid = np.array(map_size) / 2

    tr = np.array([-H[1,2], H[0,2]]) / res

    return (np.dot(H[:2,:2], (cells - mid).T).T + mid + tr).astype(int)

def transform_map(pose, grid_map):

    num_row, num_col = grid_map.shape

    J = rigid_trans_mat3([num_row/2, -num_col/2, 0], grids=True)
    H = rigid_trans_mat3(pose, grids=True)
    K = rigid_trans_mat3([-num_row/2, num_col/2, 0], grids=True)
    H = K @ H @ J

    return ndimage.affine_transform(grid_map, H, mode='nearest')

def to_twos_comp_2(val):

    """
    Returns the two's complement value corresponding to a signed integer
    value as a two-tuple. The first element of the tuple is the high byte,
    and the second element is the low byte.
    """

    if val < 0:
        val = val + (1 << 16)
    return ((val >> 8) & 0xff, val & 0xff)

def from_twos_comp_to_signed_int(val, byte=2):

    """
    Returns the signed integer value corresponding to a two's complment
    n-byte binary (the default is 2).
    """

    range_max = int((2 ** (byte * 8)) / 2)
    ones = 2 ** (byte * 8) - 1

    if val > range_max:
        val = (val ^ ones) + 1
        return -1 * val
    return val

def from_binary_to_unsigned_int(val):

    pass

def msec_to_mmsec(val):

    """
    Simply convert m/s to mm/s.
    """

    return val * 1000.0

def mm_to_cm(val):

    """
    Simply convert mm to cm.
    """

    return val * 0.1

def cm_to_mm(val):

    """
    Simply convert cm to mm.
    """

    return val * 10.0

def cap(val, smallest, highest):

    """
    Clamp a value between a smallest and a highest value (inclusive).
    """

    if val < smallest:
        return smallest
    elif val > highest:
        return highest
    return val

def is_in_circle(center, radius, position):

    """
    Check if a point at a given position is located within a circle.

    @param center:
        The center position of the circle. A length-2 tuple.
    @param radius:
        The radius of the circle.
    @param position:
        A length-2 tuple of the position to be tested.
    """

    d = math.sqrt(((center[0] - position[0]) ** 2) +\
                   (center[1] - position[1]) ** 2)

    return d <= radius

def angle_between_vectors(vec_a, vec_b):

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return math.acos(
        np.dot(vec_a, vec_b) / (norm_a * norm_b))

def angle_to_dir(angle):

    """
    Returns the direction vector from origin for the given angle in radians.
    """

    x = math.cos(angle)
    y = math.sin(angle)
    return np.array([x, y])

def disp_to_angle(disp):
    x, y = disp
    a = math.atan2(disp[1], disp[0])

    # a is in 3rd or 4th quadrant.
    if (x <= 0 and y <= 0) or (x >= 0 and y <= 0):
        a = 2 * math.pi - abs(a)

    return a

def direction_vector(p0, p1):

    direction = p1 - p0
    direction = direction / np.linalg.norm(direction)

    return direction

def angle_error(vec_from, vec_to):

    # Angle magnitude.
    h_err = angle_between_vectors(vec_from, vec_to)

    # Angle direction.
    cross_prod = np.cross(vec_from, vec_to)
    angle_dir = 0
    if cross_prod > 0:
        angle_dir = 1 # Counterclockwise.
    elif cross_prod < 0:
        angle_dir = -1 # Clockwise.

    return h_err * angle_dir

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

    # Avoid undefined ln(0) value.
    if p < 1e-10:
        p = 1e-10

    # Avoid division by 0.
    if abs(1.0 - p) < 1e-6:
        p = 0.999999

    return math.log(p / (1.0 - p))

def manhattan_distance(a, b):

    x0, y0 = a
    x1, y1 = b

    return abs(x0 - x1) + abs(x1 - y1)

def euclidean_distance(a, b):

    x0, y0 = a
    x1, y1 = b

    return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)

def morph_map(m):

    return cv2.morphologyEx(m, cv2.MORPH_OPEN, (5, 5), iterations=2)

def min_max_normalize(val, min_val, max_val):

    r = max_val - min_val
    return (val - min_val) / r

def now_file_name(postfix=''):

    now = datetime.datetime.now()
    year, month, day, hr, mn = now.year, now.month, now.day, now.hour,\
        now.minute
    filename = "{:04d}{:02d}{:02d}_{:02d}{:02d}{}".format(\
        year, month, day, hr, mn, postfix)

    return filename

def save_npy(img, postfix=''):

    filename = now_file_name(postfix)
    np.save('./saves/npy/' + filename + '.npy', img)
    print(filename, 'saved.')

def save_img(img, postfix=''):

    filename = now_file_name(postfix)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite('./saves/img/' + filename + '.png', img)
    print(filename, 'saved.')

def write_data_f64(path, data):

    if type(data) == list:
        data = np.array(data)
    if type(data) == np.ndarray and len(data.shape) == 1:
        data = np.reshape(data, (-1, 2))

    if not os.path.exists(path):

        f = tables.open_file(path, mode='w')
        atom = tables.Float64Atom()

        # Create enlargable array.
        arr = f.create_earray(f.root, 'data',\
            atom=atom, shape=(0, data.shape[1]))
        
        arr.append(data)

    else:

        f = tables.open_file(path, mode='a')
        f.root.data.append(data)

    f.close()

def read_data(path):

    f = tables.open_file(path, mode='r')
    data = f.root.data[:,:]
    f.close()
    return data

vec_log_odds_to_prob = np.vectorize(log_odds_to_prob)
vec_prob_to_log_odds = np.vectorize(prob_to_log_odds)
