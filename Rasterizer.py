import numpy as np
import scipy
from scipy import signal
from scipy import sparse
import Vectorizer


BZ_DEG = 3
BZ_CTRL_PTS = 4
DIMS = 2
MIN_OPACITY = 0.2


def curve_partitions(bezier_control_points):
    """
    Sums up the Euclidean distances between two sequential Bezier control points. Since the Bezier curve preserves
    the convex attribute (the curve itself is always inside the polygon shape defined by the 4 Bezier control
    points) it's length is bounded from above by the Euclidean distance between each sequential control points. This
    sum is the number of partitions to divide the curve to. This way the same pixels may repeat but gaps in the line
    are prevented. The calculation is preformed with matrix multiplication as follows:
    p0x p1x p2x p3x     @       1   0   0       =       p0x-p1x   p1x-p2x   p2x-p3x
    p0y p1y p2y p3y             -1  1   0               p0y-p1y   p1y-p2y   p2y-p3y
                                0   -1  1
                                0   0   -1
    At the end we raise the matrix entries by the power of 2, sum along the 1 axis, calculate the root and sum again.
    :param bezier_control_points: A numpy array with shape (2, 4).
    :return: The number of partitions to divide the curve with dtype=np.float64.
    """
    arithmetic_mat = np.eye(BZ_CTRL_PTS, BZ_DEG) - np.eye(BZ_CTRL_PTS, BZ_DEG, -1)  # TODO: Check for more pythonic way to make more efficient.
    dist_mat = np.abs(bezier_control_points.T @ arithmetic_mat)
    n = np.sum(np.sqrt(np.sum(np.square(dist_mat), axis=1)))
    return n


def t_sparse_vec(t_orig):
    """
    Given an array of n+1 values (t0, t1, ..., tn) distributed uniformly in the closed interval [0, 1] a sparse
    vector is computed with shape (4 * (n+1),) where each 4 subsequent entries are in the form of 1, ti, ti**2,
    ti**3 where i runs from 0 to n. For explanation on implementation details see the notebook.
    :param t_orig: A numpy array with shape (n+1,).
    :return: A Numpy array with shape (4*(n+1),) of the partition entries powered to their index mod 4 as follows:
    [1, t0, t0^2, t0^3, 1, t1, t1^2, t1^3, ... , 1, tn, tn^2, tn^3].
    """
    t_vec_first = np.insert(t_orig.reshape((t_orig.size, 1)), [0, 1, 1], 1, axis=1).flatten()  # [1, t, 1, 1]
    t_vec_second = np.roll(t_vec_first, 1)  # [1, 1, t, 1]
    t_vec_third = np.roll(t_vec_second, 1)  # [1, 1, 1, t]
    t_vec = t_vec_first * (t_vec_second ** 2) * (t_vec_third ** 3)
    return t_vec


def bezier_mat_by_control_points(bezier_control_points):
    """
    Evaluates the result of the Bezier matrix multiplied by the Bezier control points in order to multiply it by the
    t sparse vector later on. The matrices multiplication is as follows:
    p0x p1x p2x p3x     @       1   -3  3   -1
    p0y p1y p2y p3y             0   3   -6  3
                                0   0   3   -3
                                0   0   0   1
    :param bezier_control_points: A Numpy array with shape (4,2) containing the coordinates of the Bezier control
    points.
    :return: A Numpy array with shape (2, 4) containing the product of the above matrices.
    """
    bezier_mat = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    bez_mat_cont_pts = bezier_control_points.T @ bezier_mat
    return bez_mat_cont_pts


def bezier_curve_points(bezier_control_points):
    """
    Calculates the coordinates of the pixels representing the Bezier curve. The algorithm is based on/developed from
    Bernstien's polynomials and implemented in a matrix notation.
    :param bezier_control_points: A Numpy array with shape (4, 2) consisting of pairs of np.float64 entries where the
    first entry in each pair represents the x coordinate and the second entry represents the y coordinate of the i'th
    point. The entries can be fractional (steps of 0.5) since fractional control points improves accuracy. The result
    is rounded for whole (integers) pixel coordinates.
    :return: A Numpy array with shape (n+1, 2) which represents the vector of points to draw on the canvas where each
    point is a pair of x coordinate and y coordinate.
    """
    # t_vec_basic = np.array([1, t, t**2, t**3])  TODO: Use sparse matrix for performance optimization (scipy.sparse).
    n = curve_partitions(bezier_control_points)
    pts_num = np.uint16(n + 1)
    t_orig = np.arange(pts_num) / n  # Shape=(n+1,) TODO: Check np.linspace(s, e, n, e_ex=True, ...). Might be faster.
    t_vec = t_sparse_vec(t_orig)  # shape (4*(n+1),).
    bez_mat_ctr_p = bezier_mat_by_control_points(bezier_control_points)  # shape=(2, 4).
    bez_mat_ctr_p_blks = np.repeat(bez_mat_ctr_p.reshape(1, 2, 4), pts_num, axis=0)  # .reshape((2*(n.astype(np.uint8) + 1), t_vec.size))
    bez_mat_ctr_p_sparse = scipy.sparse.block_diag(bez_mat_ctr_p_blks)
    pixels = np.round((bez_mat_ctr_p_sparse @ t_vec).reshape((pts_num, 2)))  # TODO: Move the np.round to the caller.
    return pixels


def draw_circle(r):
    d = 2 * r - 1
    stroke = np.ones((d, d))
    for x in range(r):
        for y in range(r):
            dist = np.round(np.sqrt(x ** 2 + y ** 2))
            if dist > r:
                stroke[x][y] = 0
    stroke *= stroke[::-1]
    stroke *= stroke[::, ::-1]
    return stroke


def apply_random_texture(stroke):
    return np.random.randint(11, size=stroke.shape) * 0.1 * stroke


def apply_radius_texture(r, stroke):
    m = np.linspace(-2 * r - 1, 2 * (r + 1), 4 * (r + 1) - 1)
    x, y = np.meshgrid(m, m)
    f = (np.abs(x + y) % stroke.shape[0] + 1) / stroke.shape[0]
    # f = 1 - ((np.abs(x + y) % r) / (r - 1))
    return stroke * f


def apply_chalk_texture(p, grad, r, stroke):
    m = np.linspace(-2 * r - 1, 2 * (r + 1), 4 * (r + 1) - 1)
    x, y = np.meshgrid(m, m)
    f = (np.sum(p) * np.ones(stroke.shape)) % np.abs(1 + x + y) == r
    return stroke * f


def add_texture(p, grad, r, stroke, texture='solid'):
    # solid, chalk, charcoal, watercolour, oil_dry, oil_wet, pen, pencil, perlin_noise, splash, spark, radius_division.
    if texture == 'chalk':
        return apply_chalk_texture(p, grad, r, stroke)
    elif texture == 'radius_division':
        return apply_radius_texture(r, stroke)
    elif texture == 'random':
        return apply_random_texture(stroke)
    return stroke


def pixel_stroke(p, grad, r, shape='circle', texture='solid', blur_kernel=3, opacity=1):
    # Creating basic stroke shape.
    stroke_diameter = 4 * r + 1
    stroke = np.zeros((stroke_diameter, stroke_diameter))
    if shape == 'circle':
        stroke[r + 1:3 * r, r + 1:3 * r] = draw_circle(r)
    if shape == 'square':
        d = 2 * r - 1
        stroke[r + 1:3 * r, r + 1:3 * r] = np.ones((d, d))
    stroke = scipy.signal.convolve2d(stroke, Vectorizer.gaussian_kernel(blur_kernel), mode='same')
    # Adding texture.
    stroke = add_texture(p, grad, stroke_diameter, stroke, texture)
    # Applying opacity + considering interpolation.
    stroke *= (opacity * (1 - np.linalg.norm(p - np.round(p))))
    return stroke


def stroke_radius(radius_min, radius_max, width_style, n, i):
    # radius_style: log, root, linear, uniform.
    d = i
    c = 0
    if 2 * i > n:
        d = n - i
    if width_style == 'log':  # TODO: Replace with switch statement - better performance with numerous cases.
        c = np.log2(1 + (2 * d / n))
    elif width_style == 'root':
        c = np.sqrt(2 * d / n)
    elif width_style == 'linear':
        c = 2 * d / n
    return np.uint16(np.round(radius_min + c * (radius_max - radius_min)))


def stroke_strength(strength_style, n, i):
    d = i
    s = 1
    if 2 * i > n:
        d = n - i
    if strength_style == 'log':
        s = np.log2(1 + 2 * d / n)
    elif strength_style == 'root':
        s = np.sqrt(2 * d / n)
    elif strength_style == 'linear':
        s = 2 * d / n
    return MIN_OPACITY + (1 - MIN_OPACITY) * s


def stroke_rasterizer(bezier_control_points, radius_min=1, radius_max=5, radius_style='log', shape='circle',
                      texture='solid', blur_kernel=3, strength_style='log', canvas_shape=(1080, 1920)):
    big_canvas = np.zeros(tuple(2 * np.asarray(canvas_shape)))
    original_zero_x = np.uint16(0.5 * canvas_shape[0])
    original_zero_y = np.uint16(0.5 * canvas_shape[1])
    original_end_x = 3 * original_zero_x
    original_end_y = 3 * original_zero_y
    bzr_pts = bezier_curve_points(bezier_control_points)
    n = len(bzr_pts)
    for i in range(n):
        p = bzr_pts[i]
        grad = (bzr_pts[i + 1] - p) if (i + 1) < n else (bzr_pts[i - 1] - p)
        r = stroke_radius(radius_min, radius_max, radius_style, n, i)
        s = stroke_strength(strength_style, n, i)
        stroke = pixel_stroke(p, grad, r, shape, texture, blur_kernel, s)
        # Placing the stroke on the canvas.
        r_s = stroke.shape[0]//2
        new_p_x = np.uint16(p[0]) + original_zero_x
        new_p_y = np.uint16(p[1]) + original_zero_y
        big_canvas[new_p_x - r_s:new_p_x + r_s + 1, new_p_y - r_s:new_p_y + r_s + 1] += stroke
    canvas = big_canvas[original_zero_x:original_end_x, original_zero_y:original_end_y]
    canvas = np.clip(canvas, 0, 1)
    return canvas


def bezier_curve_rasterizer(bezier_control_points, stroke_width=1, texture=None, canvas_shape=(1080, 1920)):
    bzr_pts = bezier_curve_points(bezier_control_points)
    bsc_pxls = scipy.sparse.csr_matrix((np.ones(len(bzr_pts)), (bzr_pts.T[0], bzr_pts.T[1])), shape=canvas_shape).toarray()
    return bsc_pxls


# ------------------------------------------------ Graveyard Below -----------------------------------------------------

# def bezier_curve_point(bezier_control_points, t):
#     bezier_mat = np.array([[1, -3, -3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
#     t_vec = np.array([1, t, t**2, t**3])
#     point = np.matmul(np.matmul(bezier_control_points, bezier_mat), t_vec)
#     return point