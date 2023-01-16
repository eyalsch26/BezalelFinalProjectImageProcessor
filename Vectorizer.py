import numpy as np
import scipy.signal
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import Colourizer
import Rasterizer


GAUSSIAN_KERNEL = 9
HARRIS_W = 5
GRAD_DIRECTIONS_NUM = 4
QUANTIZE_DEGREE_STEP = 45
MAX_CURVE_TURNS = 5


def gaussian_kernel(kernel_size):
    """
    Constructs a square symmetric 2D Gaussian kernel according to the given kernel size. The kernel is normalized.
    :param kernel_size: An integer representing the size of the kernel.
    :return: A 2D numpy array of the gaussian kernel. For example, for kernel_size=3 the function returns
    the normalized kernel: [[1,2,1],[2,4,2],[1,2,1]]/16. The entries are of type float64.
    """
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = scipy.signal.convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = scipy.signal.convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def sobel_kernel(direction='x'):
    """
    Builds a Sobel kernel in the given direction:
    For horizontal derivatives (in the x axis): [1, 0, -1]
                                                [2, 0, -2]
                                                [1, 0, -1]
    For vertical derivatives (in the y axis):   [1, 2, 1]
                                                [0, 0, 0]
                                                [-1, -2, -1].
    :param direction: The desired direction to derive. The orthogonal direction is blurred.
    :return: A numpy array with shape (3, 3): A Sobel kernel with respect to the given differentiation direction.
    The entries are of type float64.
    """
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)  # / 4.0
    if direction == 'y':
        return kernel.T
    return kernel


def laplacian_kernel():
    """
    Builds the Laplacian kernel matrix: [0, 1, 0]
                                        [1, 4, 1]
                                        [0, 1, 0]
    The Laplacian is the divergence of the gradient (sum of partial derivatives).
    :return: A numpy array with shape (3, 3): The Laplacian kernel matrix. The entries are of type float64.
    """
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def two_powers_kernel():
    """
    Builds a kernel matrix constructed in spiral form of the powers of two: [1,   2,   4]
                                                                            [128, 512, 8]
                                                                            [64,  32, 16].
    The motivation is to detect specific patterns by distinct numbers, created by convolving the above kernel with
    the image. Each pattern sums up to a distinct number.
    :return: A numpy array with shape (3, 3). The entries are of type float64.
    """
    return np.array([[1, 2, 4], [128, 512, 8], [64, 32, 16]], dtype=np.float64)


def one_center_kernel():
    """
    Builds a kernel matrix with twos on the outer border and one in the center: [2, 2, 2]
                                                                                [2, 1, 2]
                                                                                [2, 2, 2].
    The motivation is to detect the initial corners (the starting point of a line), created by convolving the above
    kernel with the image. Each initial corner (referred to as "i corner" in this project) sums up to 3 exactly.
    :return: A numpy array with shape (3, 3). The entries are of type float64.
    """
    return np.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]], dtype=np.float64)


def t_x_corners_kernel():
    """
    Builds a kernel matrix constructed in the following way: [10, 2, 10]
                                                             [2,  1, 2]
                                                             [10, 2, 10].
    The motivation is to detect the cross corners and "t junctions", created by convolving the above
    kernel with the image. Each corner (referred to as "t corner" or "x corner" in this project) sums up to 7 and 31
    or 9 and 41 respectively exactly.
    :return: A numpy array with shape (3, 3). The entries are of type float64.
    """
    return np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]], dtype=np.float64)


def kernel_padder(kernel, im_shape):
    expanded_kernel = np.zeros(im_shape, np.float64)
    top_left = int(np.round((im_shape[0] - kernel.shape[0]) * 0.5))
    top_right = top_left + kernel.shape[0]
    bottom_left = int(np.round((im_shape[1] - kernel.shape[1]) * 0.5))
    bottom_right = bottom_left + kernel.shape[1]
    expanded_kernel[top_left:top_right, bottom_left:bottom_right] = kernel
    return expanded_kernel


def dft2D(image_frame):
    image_dft = np.fft.fft2(np.float64(image_frame))
    image_dft_shifted = np.fft.fftshift(image_dft)
    return image_dft_shifted


def idft2D(fourier_image):
    fourier_image_shifted_back = np.fft.ifftshift(fourier_image)
    return np.real(np.fft.ifft2(fourier_image_shifted_back))


def blur_image(im, gaussian_kernel_size):
    # Creating the image of the kernel.
    mask = gaussian_kernel(gaussian_kernel_size)
    mask_expanded = kernel_padder(mask, im.shape)
    # Transforming to Fourier domain.
    gray_im_fourier = dft2D(im)
    mask_fourier = dft2D(mask_expanded)
    # Multiplying in Fourier domain.
    frequency_result = gray_im_fourier * np.abs(mask_fourier)
    # Back to time domain.
    im_result = idft2D(frequency_result)
    return im_result


def laplacian_image(im):
    # Creating the image of the kernel.
    mask = laplacian_kernel()
    mask_expanded = kernel_padder(mask, im.shape)
    # Transforming to Fourier domain.
    gray_im_fourier = dft2D(im)
    mask_fourier = dft2D(mask_expanded)
    # Multiplying in Fourier domain.
    frequency_result = gray_im_fourier * np.abs(mask_fourier)
    # Returning to time domain.
    im_result = idft2D(frequency_result)
    return im_result


def remove_isolated_pixels(im):
    # Creating the image of the kernel.
    mask = np.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]], dtype=np.float64)
    # Convolving to remove the isolated pixels.
    im_result = im - (scipy.signal.convolve2d(im, mask, mode='same') == 1)
    return im_result


def thin_edges(im):
    # Removing double edges. E.g.: [0,0,0],[1,1,1],[1,1,1]
    tep = scipy.signal.convolve2d(im, two_powers_kernel(), mode='same')
    im[(tep == 574) | (tep == 760)] = 0
    # Removing spikes. E.g.: [0,0,0],[0,1,1],[1,1,1]
    tep = scipy.signal.convolve2d(im, two_powers_kernel(), mode='same')
    im[(tep == 542) | (tep == 632) | (tep == 737) | (tep == 647) |
       (tep == 572) | (tep == 752) | (tep == 707) | (tep == 527)] = 0
    # Removing spikes. E.g.: [0,0,0],[0,1,0],[1,1,1]
    tep = scipy.signal.convolve2d(im, two_powers_kernel(), mode='same')
    im[(tep == 519) | (tep == 540) | (tep == 624) | (tep == 705)] = 0
    return im


def clean_undesired_pixels(im):
    clean_im = thin_edges(im)
    cleaner_im = remove_isolated_pixels(clean_im)
    return cleaner_im


def detect_edges(im, t1_co=0.975, t2_co=0.995):
    # Computing Laplacian/Sobel on the image.
    # s = sobel_gradient(im)  # Works yet not as good as just laplacian - thick lines.
    # lap_im = sobel_gradient(s[0])[0] + sobel_gradient(s[1])[1]
    lap_im = laplacian_image(im)  # For images from reality use: blur_image(im, 15)
    lap_im -= np.min(lap_im)  # Clipping to [0, 1].
    lap_im /= np.max(lap_im)  # Normalizing.
    # Computing thresholds.
    t1 = t1_co * (np.mean(lap_im) + np.std(lap_im))
    t2 = t2_co * t1
    # Result according to the thresholds.
    edges_im = np.zeros(im.shape)
    edges_im[lap_im < t2] = 1
    edges_im[lap_im >= t1] = 0
    weak_edge_mask = (lap_im >= t2) & (lap_im < t1)  # Detect weak edges which touch strong edges.
    weak_edges = maximum_filter(edges_im, footprint=np.ones((3, 3))) * weak_edge_mask
    edges_im += weak_edges
    # Remove isolated pixels and spikes pixels.
    edges_im = clean_undesired_pixels(edges_im)
    return edges_im


def corner_gradient_kernels():
    hrz = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    vtc = hrz.T
    dsc = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]])
    acc = dsc[::-1]
    return np.array([hrz, vtc, dsc, acc])


def detect_corners_0(edges_im):
    grd_krn = corner_gradient_kernels()
    crn_grd = np.array([scipy.signal.convolve2d(edges_im, grd_krn[i], mode='same') for i in range(4)]) * edges_im
    # Creating the basic gradients. Four images to be multiply with the filtered corners.
    s_e = np.ones((edges_im.shape[0], edges_im.shape[1], 2))
    n_e = s_e * np.array([1, -1])
    e = s_e * np.array([1, 0])
    s = s_e * np.array([0, 1])
    bsc_grd = np.array([e, s, s_e, n_e])  # Stacking in one array.
    # Multiplying the basic gradients with the filtered ones to get the orientation of the gradient in each pixel.
    grd_spr = np.array([bsc_grd[i] * np.expand_dims(crn_grd[i], axis=2) for i in range(GRAD_DIRECTIONS_NUM)])
    sum_grd = np.sum(grd_spr, axis=0)
    mgt_grd = np.linalg.norm(sum_grd, axis=2)
    corners = np.zeros(mgt_grd.shape)
    corners[(mgt_grd > 1) & (mgt_grd <= 3)] = 1
    i_corners = scipy.signal.convolve2d(edges_im, one_center_kernel(), mode='same') == 3
    t_x_crn_flt = scipy.signal.convolve2d(edges_im, t_x_corners_kernel(), mode='same')
    t_x_corners = np.zeros(edges_im.shape)
    t_x_corners[(t_x_crn_flt == 7) | (t_x_crn_flt == 9) | (t_x_crn_flt == 41)] = 1
    y_crn = scipy.signal.convolve2d(edges_im, two_powers_kernel(), mode='same')
    y_corners = np.zeros(edges_im.shape)
    y_corners[(y_crn == 549) | (y_crn == 660) | (y_crn == 594) | (y_crn == 585) |
              (y_crn == 586) | (y_crn == 553) | (y_crn == 676) | (y_crn == 658)] = 1
    bolt_crn = np.zeros(edges_im.shape)
    bolt_crn[(y_crn == 556) | (y_crn == 688) | (y_crn == 673) | (y_crn == 618) |
             (y_crn == 706) | (y_crn == 523) | (y_crn == 538) | (y_crn == 646)] = 1
    # c_corners = np.zeros(edges_im.shape)
    # c_corners[(y_crn == 517) | (y_crn == 532) | (y_crn == 592) | (y_crn == 577)] = 1
    corners += i_corners + t_x_corners + y_corners - bolt_crn
    return corners
    # l_corners = np.ones(edges_im.shape)[(tmp == 522) | (tmp == 552) | (tmp == 672) | (tmp == 642)]
    # r_corners = np.ones(edges_im.shape)[(tmp == 526) | (tmp == 568) | (tmp == 736) | (tmp == 643)] - c_corners - l_corners
    # x_corners = np.ones(edges_im.shape)[(tmp == 682) | (tmp == 597)]


def detect_corners(edges_im):
    corners = np.zeros(edges_im.shape)
    i_corners = scipy.signal.convolve2d(edges_im, one_center_kernel(), mode='same') == 3
    tmp = scipy.signal.convolve2d(edges_im, two_powers_kernel(), mode='same')
    c_corners = np.zeros(edges_im.shape)
    c_corners[(tmp == 517) | (tmp == 532) | (tmp == 592) | (tmp == 577)] = 1
    l_corners = np.zeros(edges_im.shape)
    l_corners[(tmp == 522) | (tmp == 552) | (tmp == 672) | (tmp == 642)] = 1
    r_corners = np.zeros(edges_im.shape)
    r_corners[(tmp == 518) | (tmp == 524) | (tmp == 536) | (tmp == 560) |
              (tmp == 608) | (tmp == 704) | (tmp == 641) | (tmp == 515)] = 1
    t_x_crn_flt = scipy.signal.convolve2d(edges_im, t_x_corners_kernel(), mode='same')
    t_x_corners = np.zeros(edges_im.shape)
    t_x_corners[(t_x_crn_flt == 7) | (t_x_crn_flt == 9) | (t_x_crn_flt == 31) | (t_x_crn_flt == 41)] = 1
    y_corners = np.zeros(edges_im.shape)
    y_corners[(tmp == 549) | (tmp == 660) | (tmp == 594) | (tmp == 585) |
              (tmp == 586) | (tmp == 553) | (tmp == 676) | (tmp == 658)] = 1
    corners += i_corners + c_corners + l_corners + r_corners + t_x_corners + y_corners
    return corners


def check_point_in_bounds(point, shape, row_init=0, column_init=0):
    """
    Verifies that the given point's coordinates are within the boundaries of the given shape.
    :param point: A numpy array with shape (, 2). The entries are of type float64.  TODO: Make sure the shape is true.
    :param shape: A tuple with shape (, 2). For example (1080, 1920).
    :param row_init: An integer indicating the starting row of the boundaries.
    :param column_init: An integer indicating the starting column of the boundaries.
    :return: Boolean. True if the point's coordinates are within the given shape and False otherwise.
    """
    row_in_bounds = point[0] >= row_init & point[0] < shape[0]
    column_in_bounds = point[1] >= column_init & point[1] < shape[1]
    return row_in_bounds & column_in_bounds


def neighborhood_bounds(pxl, shape, r_d=1, c_d=1):
    row_s = pxl[0] - r_d if pxl[0] > 0 else 0
    row_e = pxl[0] + r_d + 1 if pxl[0] < shape[0] - r_d else shape[0]
    column_s = pxl[1] - c_d if pxl[1] > 0 else 0
    column_e = pxl[1] + c_d + 1 if pxl[1] < shape[1] - c_d else shape[1]
    return row_s, row_e, column_s, column_e


def estimate_ctrl_p_1(edges_im, ctrl_p_0):
    """
    Estimates the position of the second Bezier control point. One of the properties of Bezier curves is that the
    first and last control points tangent to the line formed by the first and second (or third and last respectively)
    control points. Thus, a rough estimation of the second and third control points will be along the lines described
    above. Boundaries are checked. A corner or an empty pixel indicates a possible position for the control point.
    :param edges_im: A numpy array with shape (720, 1280). The entries are of type float64. Represents the edges
    only (with value of 1 and 0 otherwise). Corners are not included/shown in the edges_im.
    :param ctrl_p_0: A numpy array with shape (, 2). The entries are of type float64. Represents the initial point to
    start calculating the Bezier curve from.
    :return: A tuple of numpy arrays with shape (, 2) where each numpy array with shape (, 2). The entries are of type
    float64. The first array represents the estimation of the second Bezier control point and the second represents
    the vector with the direction from the first control point to the second one.
    """
    row_s, row_e, column_s, column_e = neighborhood_bounds(ctrl_p_0, edges_im.shape)
    vec_p_1 = np.argmax(edges_im[row_s:row_e, column_s:column_e])
    vec_p_1 = np.asarray(np.unravel_index(vec_p_1, (row_e - row_s + 1, column_e - column_s + 1))) - np.array([1, 1])
    ctrl_p_1 = vec_p_1 + ctrl_p_0
    in_bounds = check_point_in_bounds(ctrl_p_1 + vec_p_1, edges_im.shape)
    while in_bounds:
        ctrl_p_1 += vec_p_1
        # Checks if the current pixel is an edge and not 0 or corner.
        pxl_is_edge = edges_im[ctrl_p_1[0]][ctrl_p_1[1]] > 0
        if not pxl_is_edge:  # If the pixel is a corner or 0.
            break
        in_bounds = check_point_in_bounds(ctrl_p_1 + vec_p_1, edges_im.shape)
    return ctrl_p_1, vec_p_1


def estimate_ctrl_p_2(edges_im, next_pxl, p3_p2_vec):
    ctrl_p_2 = next_pxl + p3_p2_vec
    in_bounds = check_point_in_bounds(ctrl_p_2 + p3_p2_vec, edges_im.shape)
    while in_bounds:
        ctrl_p_2 += p3_p2_vec
        # Checks if the current pixel is an edge and not 0 or corner.
        pxl_is_edge = edges_im[ctrl_p_2[0]][ctrl_p_2[1]] > 0
        if not pxl_is_edge:  # If the pixel is a corner or 0.
            break
        in_bounds = check_point_in_bounds(ctrl_p_2 + p3_p2_vec, edges_im.shape)
    return ctrl_p_2


def find_next_pixel(edges_im, pxl):
    # Checking bounds.
    row_s, row_e, column_s, column_e = neighborhood_bounds(pxl, edges_im.shape)
    # Calculating the neighborhood of the current pixel considering the boundaries of the image.
    neighborhood = edges_im[row_s:row_e, column_s:column_e]
    # Finding the indices of the neighbors pixels which has value of 1.
    neighbors = np.argwhere(neighborhood == 1)
    if len(neighbors) == 0:  # TODO: What if there is no neighbor? Must be referred in the outer scope (calling func).
        return np.array([-1, -1])
    # Finds the nearest neighbor by calculating the minimum euclidean distance from the center of the neighborhood.
    next_pxl_vec = neighbors[np.argmin(np.linalg.norm(neighbors - np.ones((len(neighbors), 2))))] - 1
    next_pxl = pxl + next_pxl_vec
    return next_pxl


def trim_curve_im(cur_curve_im):
    coord = np.argwhere(cur_curve_im == 1)
    minima = np.min(coord, axis=0)
    row_min, column_min = minima[0], minima[1]
    maxima = np.max(coord, axis=0)
    row_max, column_max = maxima[0], maxima[1]
    trimmed = cur_curve_im[row_min:row_max + 1, column_min:column_max + 1]
    new_origin = [row_min, column_min]
    return trimmed, new_origin


def pad_trimmed_curve_im(trimmed_curve_im, padder_coefficient=3):
    x_s = trimmed_curve_im.shape[0]
    y_s = trimmed_curve_im.shape[1]
    x = padder_coefficient * x_s
    y = padder_coefficient * y_s
    padded = np.zeros((x, y))
    padded[x_s: 2 * x_s, y_s: 2 * y_s] = trimmed_curve_im
    return padded


def convert_ctrl_pts(ctrl_p_0, ctrl_p_1, ctrl_p_2, ctrl_p_3, padded_origin):
    c_p_0_t = ctrl_p_0 - padded_origin
    c_p_1_t = ctrl_p_1 - padded_origin
    c_p_2_t = ctrl_p_2 - padded_origin
    c_p_3_t = ctrl_p_3 - padded_origin
    return c_p_0_t, c_p_1_t, c_p_2_t, c_p_3_t


def find_bezier_ctrl_points(ctrl_p_0, ctrl_p_1, ctrl_p_2, ctrl_p_3, curve_im, search_step=0.5):
    cur_ctrl_p_1, cur_ctrl_p_2 = ctrl_p_1, ctrl_p_2
    bezier_control_points = np.array([ctrl_p_0, cur_ctrl_p_1, cur_ctrl_p_2, ctrl_p_3])
    cur_result = Rasterizer.bezier_curve_rasterizer(bezier_control_points, canvas_shape=curve_im.shape)
    err = curve_im - cur_result
    r_srch_rds = int(curve_im.shape[0] / 3) - 1
    c_srch_rds = int(curve_im.shape[1] / 3) - 1
    r_intervals = int(2 * (search_step ** (-1)) * r_srch_rds + 1)
    c_intervals = int(2 * (search_step ** (-1)) * c_srch_rds + 1)
    if np.min(err) < 0:  # TODO: Allow out of bounds - find a way for control points outside the frame.
        for p_1_r_ad in np.linspace(-r_srch_rds, r_srch_rds, r_intervals):
            for p_1_c_ad in np.linspace(-c_srch_rds, c_srch_rds, c_intervals):
                for p_2_r_ad in np.linspace(-r_srch_rds, r_srch_rds, r_intervals):
                    for p_2_c_ad in np.linspace(-c_srch_rds, c_srch_rds, c_intervals):
                        cur_ctrl_p_1 = ctrl_p_1 + np.array([p_1_r_ad, p_1_c_ad])
                        cur_ctrl_p_2 = ctrl_p_2 + np.array([p_2_r_ad, p_2_c_ad])
                        bezier_control_points = np.array([ctrl_p_0, cur_ctrl_p_1, cur_ctrl_p_2, ctrl_p_3])
                        cur_result = Rasterizer.bezier_curve_rasterizer(bezier_control_points,
                                                                        canvas_shape=curve_im.shape)
                        err = curve_im - cur_result
                        if np.min(err) == 0:
                            return ctrl_p_0, cur_ctrl_p_1, cur_ctrl_p_2, ctrl_p_3
    if np.min(err) < 0:
        return ctrl_p_0, ctrl_p_1, ctrl_p_2, ctrl_p_3
    return ctrl_p_0, cur_ctrl_p_1, cur_ctrl_p_2, ctrl_p_3


def trace_edge_to_bezier(edges_im, corner_im, ctrl_p_0):
    """

    :param edges_im: A numpy array with shape (720, 1280). The entries are of type float64. Represents the edges and
    corners (with value of 1 and 0 otherwise).
    :param corner_im: A numpy array with shape (720, 1280). The entries are of type float64. Represents the corners
    only (with value of 1 and 0 otherwise). Edges are not included.
    :param ctrl_p_0: A numpy array with shape (, 2). The entries are of type float64. Represents the initial point to
    start calculating the Bezier curve from.
    :return:
    """
    cur_edges_im = edges_im
    cur_curve_im = np.zeros(edges_im.shape)
    curve_len = 0
    # Estimating the location of the second bezier control point and a vector from the first to second control points.
    ctrl_p_1, p0_p1_vec = estimate_ctrl_p_1(edges_im - corner_im, ctrl_p_0)
    cur_pxl = ctrl_p_0
    # Building the current curve image until the first control point.
    while cur_pxl != ctrl_p_1:
        cur_curve_im[cur_pxl[0]][cur_pxl[1]] = 1
        cur_pxl += p0_p1_vec
        curve_len += 1
    cur_pxl = ctrl_p_1 - p0_p1_vec  # TODO: Consider at least 6 pixels.
    cur_edges_im -= cur_curve_im
    next_pxl = find_next_pixel(cur_edges_im, cur_pxl)
    while curve_len < 5:
        cur_pxl = next_pxl
        cur_curve_im[cur_pxl[0]][cur_pxl[1]] = 1
        cur_edges_im -= cur_curve_im
        next_pxl = find_next_pixel(cur_edges_im, cur_pxl)
        curve_len += 1
    p3_p2_vec = cur_pxl - next_pxl
    cur_curve_im[next_pxl[0]][next_pxl[1]] = 1
    ctrl_p_2 = estimate_ctrl_p_2(edges_im, next_pxl, p3_p2_vec)
    # Trimming the relevant window where the curve is included from the entire image.
    cur_curve_im_trim, trimmed_origin = trim_curve_im(cur_curve_im)
    # Padding the trimmed curve image to allow control points outside the frame of the original image.
    cur_curve_im_trim_pad = pad_trimmed_curve_im(cur_curve_im_trim)
    padded_origin = trimmed_origin - np.array([cur_curve_im_trim.shape[0], cur_curve_im_trim.shape[1]])
    c_p_0_t, c_p_1_t, c_p_2_t, c_p_3_t = convert_ctrl_pts(ctrl_p_0, ctrl_p_1, ctrl_p_2, next_pxl, padded_origin)
    cur_bezier_ctrl_pts = find_bezier_ctrl_points(c_p_0_t, c_p_1_t, c_p_2_t, c_p_3_t, cur_curve_im_trim_pad)


def trace_edges_to_bezier(edges_im, corner_im):
    bezier_control_points = np.zeros((1, 2, 4))
    cur_corners_im = corner_im
    cur_edges_im = 2 * edges_im - corner_im  # Edges=2. Corners=1. Empty=0.
    while np.max(cur_corners_im) > 0:
        # Defining the first bezier point.
        ctrl_p_0 = np.asarray(np.unravel_index(np.argmax(cur_corners_im), corner_im.shape))
        # Computing the second bezier point. Might be adjusted later.
        vec_p_1 = np.asarray(np.unravel_index(np.argmax(cur_edges_im[ctrl_p_0[0]-1:ctrl_p_0[0]+1,
                                                        ctrl_p_0[1]-1:ctrl_p_0[1]+1]), (3, 3))) - np.array([1, 1])
        ctrl_p_1 = 2 * vec_p_1 + ctrl_p_0  # TODO: Index out of bounds.
        while cur_edges_im[ctrl_p_1[0]][ctrl_p_1[1]] > 0:  # TODO: Index out of bounds.
            ctrl_p_1 += vec_p_1
        turns_num = 1

        corners_num = np.sum(cur_corners_im)
        cur_curve = np.zeros((3, 3))
        # Updating the current corner image by removing isolated corners.

        cur_corners_im *= remove_isolated_pixels(cur_edges_im)
    return bezier_control_points


def pair_corners(edges_im, corners_im):
    pass


def vectorize_image(im):
    edges_im = detect_edges(im)
    corners_im = detect_corners(edges_im)
    corners_pairs = pair_corners(edges_im, corners_im)
    return corners_im
    # return 0.5 * (corners_im + edges_im)
    # p = np.random.randint(1, 11) / 10  # For showreel
    # return p * (corners_im + edges_im)  # For showreel
