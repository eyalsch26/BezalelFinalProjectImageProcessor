import numpy as np
import scipy
from scipy import sparse


BZ_DEG = 3
BZ_CTRL_PTS = 4
DIMS = 2


def curve_partitions(bezier_control_points):
    """
    Sums up the Manhattan distances between two sequential Bezier control points. Since the Bezier curve preserves
    the convex attribute (the curve itself is always inside the polygon shape defined by the 4 Bezier control
    points) it's length is bounded from above by the Manhattan distance between each sequential control points. This
    sum is the number of partitions to divide the curve to. This way the same pixels may repeat but gaps in the line
    are prevented. The calculation is preformed with matrix multiplication as follows:
    p0x p1x p2x p3x     @       1   0   0       =       p0x-p1x   p1x-p2x   p2x-p3x
    p0y p1y p2y p3y             -1  1   0               p0y-p1y   p1y-p2y   p2y-p3y
                                0   -1  1
                                0   0   -1
    At the end we sum all the matrix entries.
    :param bezier_control_points: A numpy array with shape (2, 4).
    :return: The number of partitions to divide the curve with dtype=np.float64.
    """
    arithmetic_mat = np.eye(BZ_CTRL_PTS, BZ_DEG) - np.eye(BZ_CTRL_PTS, BZ_DEG, -1)  # TODO: Check for more pythonic way to make more efficient.
    dist_mat = np.abs(bezier_control_points.T @ arithmetic_mat)
    n = np.array([np.sum(dist_mat)])[0]  # TODO: Check for simpler way to return a scalar. Otherwise gives warning.
    return n


def t_sparse_vec(t_orig, t_orig_len):
    """
    Given an array of n+1 values (t0, t1, ..., tn) distributed uniformly in the closed interval [0, 1] a sparse
    vector is computed with shape (4 * (n+1),) where each 4 subsequent entries are in the form of 1, ti, ti**2,
    ti**3 where i runs from 0 to n. For explanation on implementation details see the notebook.
    :param t_orig: A numpy array with shape (n+1,).
    :param t_orig_len: An integer (np.uint8) which represents the length of the t_orig. Actually its value is n+1.
    :return: A Numpy array with shape (4*(n+1),) of the partition entries powered to their index mod 4 as follows:
    [1, t0, t0^2, t0^3, 1, t1, t1^2, t1^3, ... , 1, tn, tn^2, tn^3].
    """
    t_vec_first = np.insert(t_orig.reshape((t_orig.size, 1)), [0, 1, 1], 1, axis=1).flatten()  # [1, t, 1, 1]
    t_vec_second = np.roll(t_vec_first, 1)  # [1, 1, t, 1]
    t_vec_third = np.roll(t_vec_second, 1)  # [1, 1, 1, t]
    t_vec = t_vec_first * t_vec_second ** 2 * t_vec_third ** 3
    return t_vec


def bezier_mat_by_control_points(bezier_control_points):
    """
    Evaluates the result of the Bezier matrix multiplied by the Bezier control points in order to multiply it by the
    t sparse vector later on. The matrices multiplication is as follows:
    p0x p1x p2x p3x     @       1   -3  -3  -1
    p0y p1y p2y p3y             0   3   -6  3
                                0   0   3   -3
                                0   0   0   1
    :param bezier_control_points: A Numpy array with shape (8,) containing the coordinates of the Bezier control points.
    :return: A Numpy array with shape (1, 2, 4) containing the product of the above matrices.
    """
    bezier_mat = np.array([[1, -3, -3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    bez_mat_cont_pts = bezier_control_points.reshape(1, 2, 4) @ bezier_mat
    return bez_mat_cont_pts


def bezier_curve_points(bezier_control_points):
    """
    Calculates the coordinates of the pixels representing the Bezier curve. The algorithm is based on/developed from
    Bernstien's polynomials and implemented in a matrix notation.
    :param bezier_control_points: A Numpy array with shape (4, 2) consisting of pairs of np.float64 entries where the
    first entry in each pair represents the x coordinate and the second entry represents the y coordinate of the i'th
    point.
    :return: A Numpy array with shape (n+1, 2) which represents the vector of points to draw on the canvas where each
    point is a pair of x coordinate and y coordinate.
    """
    # t_vec_basic = np.array([1, t, t**2, t**3])  TODO: Use sparse matrix for performance optimization (scipy.sparse).
    n = curve_partitions(bezier_control_points)
    t_orig = np.arange(n + 1) / n  # Shape=(n+1,) TODO: Check np.linspace(s, e, n, e_ex=True, ...). Might be faster.
    t_vec = t_sparse_vec(t_orig, np.uint8(n + 1))
    bez_mat_ctr_p = bezier_mat_by_control_points(bezier_control_points)
    bez_mat_ctr_p_blks = np.repeat(bez_mat_ctr_p, n + 1, axis=0)  # .reshape((2 * (n.astype(np.uint8) + 1), t_vec.size))
    bez_mat_ctr_p_sparse = scipy.sparse.block_diag(bez_mat_ctr_p_blks)
    pixels = (bez_mat_ctr_p_sparse @ t_vec).reshape((n + 1, 2))
    return pixels


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