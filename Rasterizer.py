import numpy as np
import math
import scipy
from scipy import signal
from scipy import sparse
import Vectorizer
import FileManager
import Colourizer


BZ_DEG = 3
BZ_CTRL_PTS = 4
DIMS = 2
MIN_OPACITY = 0.2
TXR_NUM = 13  # 0: random. 1: solid. 2: chalk. 3: charcoal. 4: watercolour. 5: oil_dry. 6: oil_wet. 7: pen. 8: pencil.
# 9: perlin_noise. 10: splash. 11: spark. 12: radius_division. 13: bubble. 14: fur. 15: cloth.


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
    return n if n > 0 else 1


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


def curve_points_min_max(bcp):
    """
    Finds the minimum and maximum row and column coordinates of the bezier control points array given.
    :param bcp: A numpy array with shape (4, 2) represents the Bezier control points and dtype np.float64.
    :return: An array with dtype np.float64 and shape (1, 4) which represents the row min, row max, column min and
    column max.
    """
    bcp_t = bcp.T
    x_min = np.min(bcp_t, axis=1)[0]
    x_max = np.max(bcp_t, axis=1)[0]
    y_min = np.min(bcp_t, axis=1)[1]
    y_max = np.max(bcp_t, axis=1)[1]
    return np.uint16([x_min, x_max, y_min, y_max])


def stroke_shape(bcp, strk_max_radius):
    x_min, x_max, y_min, y_max = curve_points_min_max(bcp)
    row_e = x_max - x_min + 2 * strk_max_radius + 1
    column_e = y_max - y_min + 2 * strk_max_radius + 1
    shape = tuple((int(row_e), int(column_e)))
    return shape


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Textures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def generate_textures_arr(l, generation_method, uniform_txr=1, given_txr_arr=(0, 1)):
    if generation_method == 'random':
        txr_arr = np.arange(l) % TXR_NUM
        return txr_arr
    elif generation_method == 'uniform':
        txr_arr = np.ones(l) * uniform_txr
        return txr_arr
    elif generation_method == 'from_arr':
        sub_arr_len = int(np.ceil(l / len(given_txr_arr)))
        txr_arr = np.dstack((np.ones(sub_arr_len) * given_txr_arr[0]))
        for i in range(1, len(txr_arr)):
            txr_arr = np.dstack((np.ones(sub_arr_len) * txr_arr[i]))
        return txr_arr
    else:  # Default.
        txr_arr = np.zeros(l)
        return txr_arr


def draw_circle(r):
    if r < 1:
        r = 1
    r_r = math.floor(r)
    x = np.linspace(-r_r, r_r, 2 * r_r + 1)
    y = np.linspace(-r_r, r_r, 2 * r_r + 1)
    xx, yy = np.meshgrid(x, y)
    stroke = np.clip(r - np.sqrt(xx ** 2 + yy ** 2), 0, 1)
    return stroke


def draw_rim(r):
    r_r = math.floor(r)
    x = np.linspace(-r_r, r_r, 2 * r_r + 1)
    y = np.linspace(-r_r, r_r, 2 * r_r + 1)
    xx, yy = np.meshgrid(x, y)
    stroke = np.clip(r - np.sqrt(xx ** 2 + yy ** 2), 0, 1)
    return stroke


def apply_random_texture(stroke):
    return np.random.randint(11, size=stroke.shape) * 0.1 * stroke


def apply_radius_texture(r, stroke):
    m = np.linspace(-2 * r - 1, 2 * (r + 1), 4 * (r + 1) - 1)
    x, y = np.meshgrid(m, m)
    f = (np.abs(x + y) % stroke.shape[0] + 1) / stroke.shape[0]
    # f = 1 - ((np.abs(x + y) % r) / (r - 1))
    return stroke * f


# Original
def apply_chalk_texture0(p, grad, r, stroke):
    m = np.linspace(-2 * r - 1, 2 * (r + 1), 4 * (r + 1) - 1)
    x, y = np.meshgrid(m, m)
    f = (np.sum(p) * np.ones(stroke.shape)) % np.abs(1 + x + y) == r
    return stroke * f


def apply_chalk_texture(p, r, stroke):
    m = np.linspace(int(-r), int(r) + 1, 2 * int(r) + 1)
    x, y = np.meshgrid(m, m)
    f = (np.sum(p) * np.random.randint(0, 2, stroke.shape)) % np.abs(1 + x + y)
    return stroke * f


def apply_pastel_texture(p, r, stroke):
    strk_a = apply_chalk_texture(p, r, stroke)
    strk_b = strk_a.T
    strk_res = Vectorizer.blur_image(np.clip(strk_a + strk_b, 0, 1), 3)
    return strk_res


def apply_cloth_texture(stroke):
    x = np.ones(stroke.shape)
    y = np.ones(stroke.shape)
    ratio = np.random.randint(1, 3)
    x[::4 * ratio] = 0
    y[::8 // ratio] = 0
    txr = Vectorizer.blur_image(x * y.T * apply_random_texture(stroke), 3)
    return stroke * txr


def apply_filament_texture(k, stroke):  # TODO: Not working.
    res_stroke = Vectorizer.blur_image(stroke, k)
    return res_stroke


def apply_sin_texture(stroke):
    strk_shape = stroke.shape
    org = [np.random.randint(0, strk_shape[0]), np.random.randint(0, strk_shape[1])]
    wave_l_r = np.random.randint(1, 13)
    style_int = np.random.randint(0, 4)
    stroke_res = stroke
    mask = stroke != 0
    if style_int == 0:
        txr_im = sin_texture_x(strk_shape, org, wave_l_r)
        stroke_res *= txr_im
    elif style_int == 1:
        txr_im = sin_texture_y(strk_shape, org, wave_l_r)
        stroke_res *= txr_im
    else:
        txr_im = sin_texture(strk_shape, org, wave_l_r)
        stroke_res *= txr_im
    stroke_res -= np.min(stroke_res)
    stroke_res /= np.max(stroke_res)
    stroke_res *= mask
    return stroke_res


def sin_texture_x(im_shape, org, wave_len_r=1):
    x = np.linspace(-org[0], im_shape[0] - org[0], im_shape[0])
    y = np.linspace(-org[1], im_shape[1] - org[1], im_shape[1])
    xx, yy = np.meshgrid(x, y)
    xx /= im_shape[0] - org[0]
    xx *= wave_len_r
    sin_im_x = np.sin(xx)
    return sin_im_x


def sin_texture_y(im_shape, org, wave_len_r=1):
    x = np.linspace(-org[0], im_shape[0] - org[0], im_shape[0])
    y = np.linspace(-org[1], im_shape[1] - org[1], im_shape[1])
    xx, yy = np.meshgrid(x, y)
    yy /= im_shape[1] - org[1]
    yy *= wave_len_r
    sin_im_y = np.sin(yy)
    return sin_im_y


def sin_texture(im_shape, org, wave_len_r=1):
    x = np.linspace(-org[0], im_shape[0] - org[0], im_shape[0])
    y = np.linspace(-org[1], im_shape[1] - org[1], im_shape[1])
    xx, yy = np.meshgrid(x, y)
    xx /= im_shape[0] - org[0]
    yy /= im_shape[1] - org[1]
    xx *= wave_len_r
    yy *= wave_len_r
    x_sin = np.sin(xx)
    y_sin = np.sin(yy)
    sin_im = x_sin + y_sin
    return sin_im


def apply_stripes_texture(stroke):
    strk_shape = stroke.shape
    stripe_w = np.random.randint(1, 2 + int(np.min(strk_shape) // 8))
    space_w = np.random.randint(1, 2 + int(np.min(strk_shape) // 8))
    direction = 'vertical' if np.random.randint(0, 2) == 0 else 'horizontal'
    style_int = np.random.randint(0, 3)
    style = 'uniform'
    if style_int == 0:
        style = 'scale'
    elif style_int == 1:
        style = 'random'
    txr_im = stripes_texture(strk_shape, stripe_w, space_w, direction, style)
    return stroke * txr_im


def stripes_texture(im_shape, stripe_width, space_width, direction='horizontal', style='uniform'):
    stripes_im = np.zeros(im_shape)
    r = im_shape[0] // (stripe_width + space_width)
    if direction == 'vertical':
        stripes_im = stripes_im.T
        r = im_shape[1] // (stripe_width + space_width)
    width_addition_arr = np.arange(stripe_width)
    cur_width_add = 0
    start = 0
    end = stripe_width
    for i in range(r):
        if style == 'scale':
            cur_width_add = width_addition_arr[i % stripe_width]
            end += cur_width_add
        elif style == 'random':
            cur_width_add = width_addition_arr[np.random.randint(0, stripe_width)]
            end += cur_width_add
        stripes_im[start:end:] = 1
        start = np.clip(end + space_width, 0, stripes_im.shape[0] - 1)  # i * (stripe_width + cur_width_add + space_width)
        end = np.clip(start + stripe_width + cur_width_add, 0, stripes_im.shape[0] - 1)  # i * (stripe_width + space_width) + stripe_width
    if direction == 'vertical':
        stripes_im = stripes_im.T
    return stripes_im


def roll_texture_image(image, roll_iter, direction='horizontal'):
    axis = 0 if direction == 'horizontal' else 1
    im = np.roll(image, roll_iter, axis)
    return im


def volume_spread(im_shape, org):
    pass


def add_texture(p, r, stroke, texture=1):
    # solid, chalk, charcoal, watercolour, oil_dry, oil_wet, pen, pencil, perlin_noise, splash, spark, radius_division.
    if texture == 0:  # Random.
        return apply_random_texture(stroke)
    elif texture == 2:  # Cloth.
        return apply_cloth_texture(stroke)
    elif texture == 3:  # Stripes.
        return apply_stripes_texture(stroke)
    elif texture == 4:  # Sin.
        return apply_sin_texture(stroke)
    elif texture == 5: # Chalk.
        return apply_chalk_texture(p, r, stroke)
    elif texture == 6: # Pastel.
        return apply_pastel_texture(p, r, stroke)
    return stroke  # Solid.


def add_texture1(stroke, texture=1):
    # solid, chalk, charcoal, watercolour, oil_dry, oil_wet, pen, pencil, perlin_noise, splash, spark, radius_division.
    if texture == 0:  # 'random'
        return apply_random_texture(stroke)
    elif texture == 15:  # 'cloth'
        return apply_cloth_texture(stroke)
    return stroke  # 'solid'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stroke ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    return radius_min + c * (radius_max - radius_min)


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


def pixel_stroke(p, r, blur_kernel=3, texture=1, opacity=1):
    # Creating basic stroke shape.
    stroke = draw_circle(r)
    # Adding blur.
    stroke = scipy.signal.convolve2d(stroke, Vectorizer.gaussian_kernel(blur_kernel), mode='same')
    # Adding texture.
    stroke = add_texture(p, r, stroke, texture)
    # Applying opacity + considering interpolation.
    stroke *= (opacity * (1 - np.linalg.norm(p - np.round(p))))
    return stroke


def pixel_stroke1(p, r, blur_kernel=3, opacity=1):
    # Creating basic stroke shape.
    stroke = draw_circle(r)
    # Adding blur.
    stroke = scipy.signal.convolve2d(stroke, Vectorizer.gaussian_kernel(blur_kernel), mode='same')
    # Applying opacity + considering interpolation.
    stroke *= (opacity * (1 - np.linalg.norm(p - np.round(p))))
    return stroke


def stroke_rasterizer(bezier_control_points, radius_min=1, radius_max=5, radius_style='log', shape='circle',
                      texture=1, blur_kernel=3, strength_style='log', canvas_shape=(1080, 1920)):
    # Preparing the image base.
    big_canvas = np.zeros(tuple(2 * np.asarray(canvas_shape)))
    original_zero_x = np.uint16(0.5 * canvas_shape[0])
    original_zero_y = np.uint16(0.5 * canvas_shape[1])
    original_end_x = 3 * original_zero_x
    original_end_y = 3 * original_zero_y
    # Computing the pixels of the curve.
    bzr_pts = bezier_curve_points(bezier_control_points)
    n = len(bzr_pts)
    # Computing each pixel stroke.
    for i in range(n):
        p = bzr_pts[i]
        # grad = (bzr_pts[i + 1] - p) if (i + 1) < n else (bzr_pts[i - 1] - p)
        r = stroke_radius(radius_min, radius_max, radius_style, n, i)
        s = stroke_strength(strength_style, n, i)
        stroke = pixel_stroke(p, r, blur_kernel, texture, s)
        # Placing the stroke on the canvas.
        r_s = stroke.shape[0]//2
        new_p_x = np.uint16(p[0]) + original_zero_x
        new_p_y = np.uint16(p[1]) + original_zero_y
        big_canvas[new_p_x - r_s:new_p_x + r_s + 1, new_p_y - r_s:new_p_y + r_s + 1] += stroke
    canvas = big_canvas[original_zero_x:original_end_x, original_zero_y:original_end_y]
    canvas = np.clip(canvas, 0, 1)
    return canvas


def stroke_rasterizer1(bcp, radius_min=1, radius_max=5, radius_style='log', shape='circle', texture=1, blur_kernel=3,
                      strength_style='log', canvas_shape=(1080, 1920)):
    # Preparing the image base.
    big_canvas_shape = tuple((int(canvas_shape[0] + 2 * radius_max + 1), int(canvas_shape[1] + 2 * radius_max + 1)))
    canvas = np.zeros(big_canvas_shape)
    # Computing the pixels of the curve.
    bzr_pts = bezier_curve_points(bcp)
    # Generating a stroke canvas.
    x_min, x_max, y_min, y_max = curve_points_min_max(bcp)
    strk_shape = stroke_shape(bcp, radius_max)
    stroke_canvas = np.zeros(strk_shape)
    n = len(bzr_pts)
    # Computing each pixel stroke.
    for i in range(n):
        p = bzr_pts[i]
        r = stroke_radius(radius_min, radius_max, radius_style, n, i)
        s = stroke_strength(strength_style, n, i)
        pxl_strk = pixel_stroke(p, r, blur_kernel, s)
        # Placing the stroke on the canvas.
        p_x_e = int(p[0] - x_min + radius_max + np.ceil(pxl_strk.shape[0]/2))
        p_x_s = p_x_e - pxl_strk.shape[0]
        p_y_e = int(p[1] - y_min + radius_max + np.ceil(pxl_strk.shape[1]/2))
        p_y_s = p_y_e - pxl_strk.shape[1]
        stroke_canvas[p_x_s:p_x_e, p_y_s:p_y_e] += pxl_strk  # r includes the center so no +1 needed at the end.
        # Adding texture.
    stroke_canvas = add_texture(stroke_canvas, texture)
    row_s = int(x_min)
    row_e = int(x_max + 2 * radius_max + 1)
    column_s = int(y_min)
    column_e = int(y_max + 2 * radius_max + 1)
    canvas[row_s:row_e, column_s:column_e] = stroke_canvas
    canvas = canvas[int(radius_max):int(-radius_max) - 1, int(radius_max):int(-radius_max) - 1]
    canvas = np.clip(canvas, 0, 1)
    return canvas


def bezier_curve_rasterizer(bezier_control_points, stroke_width=1, texture=None, canvas_shape=(1080, 1920)):
    bzr_pts = bezier_curve_points(bezier_control_points)
    bsc_pxls = scipy.sparse.csr_matrix((np.ones(len(bzr_pts)), (bzr_pts.T[0], bzr_pts.T[1])), shape=canvas_shape).toarray()
    return bsc_pxls


def bezier_curves_rasterizer(bezier_control_points_arr, canvas_shape=(1080, 1920)):
    im = np.zeros(canvas_shape)
    for i in range(len(bezier_control_points_arr)):
        cur_bzr_ctrl_pts_arr = bezier_control_points_arr[i]
        im += bezier_curve_rasterizer(cur_bzr_ctrl_pts_arr, canvas_shape=canvas_shape)  # Original
    # im /= np.max(im)  # For visualization only - indicates how many times a pixel has been coloured. Make sure no clip.
    return np.clip(im, 0, 1)


# Works.
def strokes_rasterizer(bezier_control_points_arr, radius_min=1, radius_max=5, radius_style='uniform',
                                texture='random', canvas_shape=(1080, 1920), canvas_scalar=1.5):
    im = np.zeros(canvas_shape)
    bezier_control_points_arr_scaled = canvas_scalar * bezier_control_points_arr
    for i in range(len(bezier_control_points_arr)):
        cur_bzr_ctrl_pts = bezier_control_points_arr_scaled[i]
        im += stroke_rasterizer(cur_bzr_ctrl_pts, radius_min, radius_max, 'uniform',
                                texture=0, canvas_shape=canvas_shape)
    im /= np.max(im)  # For visualization only - indicates how many times a pixel has been coloured. Make sure no clip.
    return np.clip(np.log2(im + 1), 0, 1)  # Original.

# Works.
def diminish_bcps_num(bezier_control_points_arr, min_l_r=0.5):
    l_arr = list(map(curve_partitions, bezier_control_points_arr))
    l_arr_max = np.max(l_arr)
    l_max = min_l_r * l_arr_max
    idx_arr = np.argwhere(l_arr >= l_max).flatten()
    diminished_bcps = bezier_control_points_arr[idx_arr]
    return diminished_bcps


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Content ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def content_rasterizer(im, cnvs_shape, canvas_scaler, idx_factor, displace, displace_max, radius_min, blur_kernel, texture, r_min, r_max,
                       g_min, g_max, b_min, b_max, colour_style, alpha, alpha_c, alpha_f):
    # Preparing the image.
    canvas_output_x, canvas_output_y = cnvs_shape
    im_r = np.zeros(cnvs_shape)
    im_g = np.zeros(cnvs_shape)
    im_b = np.zeros(cnvs_shape)
    im_a = np.zeros(cnvs_shape)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    alpha_im = Colourizer.alpha_input_image(im)
    im_y_alpha = im_y * alpha_im
    # Preparing the image base.
    original_zero_x = np.uint16(0.5 * cnvs_shape[0])
    original_zero_y = np.uint16(0.5 * cnvs_shape[1])
    original_end_x = 3 * original_zero_x
    original_end_y = 3 * original_zero_y
    # Computing the pixels of the curve.
    pxl_arr = np.argwhere(im_y_alpha != 0) * canvas_scaler
    pxl_num = len(pxl_arr)
    coof = 1 - (pxl_num / (cnvs_shape[0] * cnvs_shape[1]))
    content_radius = np.sqrt(0.5 * pxl_num / np.pi)
    diminish_step = int(0.5 * content_radius + 0.01 * pxl_num)
    pxl_idx_arr = np.random.randint(0, diminish_step, pxl_num)
    pxl_pts = pxl_arr[pxl_idx_arr == 0]
    center = 0.5 * (np.min(pxl_pts, axis=0) + np.max(pxl_pts, axis=0))
    radius_coof = 0.25 * coof * idx_factor
    center_vecs = pxl_pts - center  # Vectors from the center to the points.
    norms_arr = np.linalg.norm(center_vecs, axis=1)
    n = len(pxl_pts)
    if displace == 'True':
        radius_coof = 0.25 * coof * (1 - idx_factor)
        displace_arr = np.random.randint(0, displace_max, n)
        displace_vecs = idx_factor * displace_arr.reshape((len(displace_arr), 1)) * center_vecs / norms_arr.reshape(
            (len(norms_arr), 1)) + center_vecs
        pxl_pts += displace_vecs
    radius_arr = radius_coof * (np.max(norms_arr) - norms_arr) + radius_min
    rgb_range = np.array([[r_min, r_max], [g_min, g_max], [b_min, b_max]], dtype=int)
    clr_arr = Colourizer.generate_colours_arr(n, colour_style, rgb_range, idx_factor)  # Defining colour.
    txr_arr = np.random.randint(0, 5, n)
    # Computing each pixel stroke.
    for i in range(n):
        big_canvas = np.zeros(tuple(2 * np.asarray(cnvs_shape)))
        p = pxl_pts[i]
        r = radius_arr[i]
        s = r / np.max(radius_arr)
        cur_txr = txr_arr[i]
        cur_clr = clr_arr[i]
        stroke = pixel_stroke(p, r, blur_kernel, cur_txr, s)
        # Placing the stroke on the canvas.
        r_s = stroke.shape[0] // 2
        # new_p_x = np.uint16(p[0]) + original_zero_x  # Original.
        # new_p_y = np.uint16(p[1]) + original_zero_y  # Original.
        new_p_x = int(p[0]) + original_zero_x
        new_p_y = int(p[1]) + original_zero_y
        big_canvas[new_p_x - r_s:new_p_x + r_s + 1, new_p_y - r_s:new_p_y + r_s + 1] += stroke
        stroke = big_canvas[original_zero_x:original_end_x, original_zero_y:original_end_y]
        stroke = np.clip(stroke, 0, 1)
        stroke_rgb = np.repeat(stroke, Colourizer.CLR_DIM).reshape((int(canvas_output_x), int(canvas_output_y),
                                                                    Colourizer.CLR_DIM))
        stroke_rgb = Colourizer.colour_stroke(stroke_rgb, cur_clr[0], cur_clr[1], cur_clr[2])
        stroke_alpha_im = Colourizer.alpha_channel(stroke, alpha, alpha_c, int(alpha_f))
        stroke_im_r = stroke_rgb[::, ::, :1:].reshape(cnvs_shape)
        stroke_im_g = stroke_rgb[::, ::, 1:2:].reshape(cnvs_shape)
        stroke_im_b = stroke_rgb[::, ::, 2::].reshape(cnvs_shape)
        im_r = Colourizer.composite_rgb(stroke_im_r, im_r, stroke_alpha_im)
        im_g = Colourizer.composite_rgb(stroke_im_g, im_g, stroke_alpha_im)
        im_b = Colourizer.composite_rgb(stroke_im_b, im_b, stroke_alpha_im)
        im_a = Colourizer.composite_alpha(stroke_alpha_im, im_a)
    im_rgb = np.dstack((im_r, im_g, im_b))
    return im_rgb, im_a



def content_rasterizer0(im, idx=24, min_pts_num=80, diminish_pts_f=20):
    relative_idx = (idx % FileManager.FPS) / FileManager.FPS
    # Contour.
    contour = Vectorizer.detect_edges(im)
    contour_cof = np.argwhere(contour != 0)
    contour_pxl_num = len(contour_cof)
    initial_contour_pxl_num = contour_pxl_num
    contour_cof = np.roll(contour_cof, int(np.floor(relative_idx * contour_pxl_num)), axis=0)
    if contour_pxl_num > min_pts_num:
        contour_cof = contour_cof[::diminish_pts_f]
        contour_pxl_num = len(contour_cof)
    pts_num = contour_pxl_num + 4 - contour_pxl_num % 4
    bzr_crv_num = pts_num // 4
    bcp = np.repeat([contour_cof], 2, axis=0).reshape((2 * contour_pxl_num, 2))[:pts_num].reshape(bzr_crv_num, 4, 2)

    # rst_im = strokes_rasterizer(bcp)
    # # Center of mass.
    # center = np.average(contour_cof, axis=0)  # Point to collapse to.
    # # Vectors to center.
    # vecs_to_center = contour_cof - center
    # vecs_norm = np.linalg.norm(vecs_to_center, axis=1)
    # radius_arr = 0.5 * vecs_norm
    return bcp, bzr_crv_num, initial_contour_pxl_num


# ------------------------------------------------ Graveyard Below -----------------------------------------------------

# def bezier_curve_point(bezier_control_points, t):
#     bezier_mat = np.array([[1, -3, -3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
#     t_vec = np.array([1, t, t**2, t**3])
#     point = np.matmul(np.matmul(bezier_control_points, bezier_mat), t_vec)
#     return point
#
#
# def draw_circle0(r):
#     d = 2 * r - 1
#     stroke = np.ones((d, d))
#     # Constructing quarter of a circle.
#     for x in range(r):
#         for y in range(r):
#             dist = np.round(np.sqrt(x ** 2 + y ** 2))
#             if dist > r:
#                 stroke[x][y] = 0
#     # Reflecting along the vertical axis. Now we have half a circle.
#     stroke *= stroke[::-1]
#     # Reflecting along the horizontal axis. Now we have full a circle.
#     stroke *= stroke[::, ::-1]
#     return stroke
#
#
# def add_texture0(p, grad, r, stroke, texture='solid'):
#     # solid, chalk, charcoal, watercolour, oil_dry, oil_wet, pen, pencil, perlin_noise, splash, spark, radius_division.
#     if texture == 'chalk':
#         return apply_chalk_texture(p, grad, r, stroke)
#     elif texture == 'radius_division':
#         return apply_radius_texture(r, stroke)
#     elif texture == 'random':
#         return apply_random_texture(stroke)
#     return stroke
#
#
# def pixel_stroke0(p, grad, r, shape='circle', texture='solid', blur_kernel=3, opacity=1):
#     # Creating basic stroke shape.
#     stroke_diameter = 4 * r + 1
#     stroke = np.zeros((stroke_diameter, stroke_diameter))
#     if shape == 'circle':
#         stroke[r + 1:3 * r, r + 1:3 * r] = draw_circle(r)
#     if shape == 'square':
#         d = 2 * r - 1
#         stroke[r + 1:3 * r, r + 1:3 * r] = np.ones((d, d))
#     stroke = scipy.signal.convolve2d(stroke, Vectorizer.gaussian_kernel(blur_kernel), mode='same')
#     # Adding texture.
#     stroke = add_texture(p, grad, stroke_diameter, stroke, texture)
#     # Applying opacity + considering interpolation.
#     stroke *= (opacity * (1 - np.linalg.norm(p - np.round(p))))
#     return stroke
#
#
# def pixel_stroke1(p, grad, r, shape='circle', texture='solid', blur_kernel=3, opacity=1):
#     # Creating basic stroke shape.
#     stroke_diameter = np.uint16(2 * math.floor(r) + 1)
#     stroke = np.zeros((stroke_diameter, stroke_diameter))
#     if shape == 'circle':
#         stroke = draw_circle(r)
#     if shape == 'square':
#         d = 2 * r - 1
#         stroke[r + 1:3 * r, r + 1:3 * r] = np.ones((d, d))
#     stroke = scipy.signal.convolve2d(stroke, Vectorizer.gaussian_kernel(blur_kernel), mode='same')
#     # Adding texture.
#     stroke = add_texture(p, grad, stroke_diameter, stroke, texture)
#     # Applying opacity + considering interpolation.
#     stroke *= (opacity * (1 - np.linalg.norm(p - np.round(p))))
#     return stroke
#
#
# def stroke_radius0(radius_min, radius_max, width_style, n, i):
#     # radius_style: log, root, linear, uniform.
#     d = i
#     c = 0
#     if 2 * i > n:
#         d = n - i
#     if width_style == 'log':  # TODO: Replace with switch statement - better performance with numerous cases.
#         c = np.log2(1 + (2 * d / n))
#     elif width_style == 'root':
#         c = np.sqrt(2 * d / n)
#     elif width_style == 'linear':
#         c = 2 * d / n
#     return np.uint16(np.round(radius_min + c * (radius_max - radius_min)))
#
#
# def stroke_rasterizer0(bezier_control_points, radius_min=1, radius_max=5, radius_style='log', shape='circle',
#                       texture='solid', blur_kernel=3, strength_style='log', canvas_shape=(1080, 1920)):
#     # Preparing the image base.
#     big_canvas = np.zeros(tuple(2 * np.asarray(canvas_shape)))
#     original_zero_x = np.uint16(0.5 * canvas_shape[0])
#     original_zero_y = np.uint16(0.5 * canvas_shape[1])
#     original_end_x = 3 * original_zero_x
#     original_end_y = 3 * original_zero_y
#     bzr_pts = bezier_curve_points(bezier_control_points)
#     n = len(bzr_pts)
#     # Computing each pixel stroke.
#     for i in range(n):
#         p = bzr_pts[i]
#         grad = (bzr_pts[i + 1] - p) if (i + 1) < n else (bzr_pts[i - 1] - p)
#         r = stroke_radius(radius_min, radius_max, radius_style, n, i)
#         s = stroke_strength(strength_style, n, i)
#         stroke = pixel_stroke(p, grad, r, shape, texture, blur_kernel, s)
#         # Placing the stroke on the canvas.
#         r_s = stroke.shape[0]//2
#         new_p_x = np.uint16(p[0]) + original_zero_x
#         new_p_y = np.uint16(p[1]) + original_zero_y
#         big_canvas[new_p_x - r_s:new_p_x + r_s + 1, new_p_y - r_s:new_p_y + r_s + 1] += stroke
#     canvas = big_canvas[original_zero_x:original_end_x, original_zero_y:original_end_y]
#     canvas = np.clip(canvas, 0, 1)
#     return canvas



