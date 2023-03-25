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


def find_neighbours(edges_im, pxl):
    # Checking bounds.
    row_s, row_e, column_s, column_e = neighborhood_bounds(pxl, edges_im.shape)
    # Calculating the neighborhood of the current pixel considering the boundaries of the image.
    neighborhood = np.copy(edges_im[row_s:row_e, column_s:column_e])
    # Deleting the origin pixel (a pixel is not a neighbour of itself).
    neighborhood[1][1] = 0
    # Finding the indices of the neighbors pixels which has value of 1.
    neighbours = np.argwhere(neighborhood == 1)
    return neighbours


def vectors_to_neighbours(neighbours):
    return neighbours - 1


def sort_neighbours_by_angle(origin_vec, neighbours, neighbours_vecs):
    neighbours_vecs_normalized = neighbours_vecs / np.linalg.norm(neighbours_vecs)
    # origin_vecs_normalized = np.repeat(origin_vec / np.linalg.norm(origin_vec), len(neighbours)).reshape(
    origin_vec_normalized = origin_vec / np.linalg.norm(origin_vec)
    angles_arr = np.array([])
    for i in range(len(neighbours)):
        angle = np.arccos(np.clip(np.dot(origin_vec_normalized, neighbours_vecs_normalized[i]), -1.0, 1.0)) * 180 / np.pi
        angles_arr = np.append(angles_arr, angle)
    # angles_arr = np.arccos(np.clip(np.linalg.multi_dot([origin_vecs_normalized, neighbours_vecs_normalized]), -1.0,
    #                                1.0)) * 180 / np.pi  # Dimensions error.
    neighbours_sorted = neighbours[np.argsort(angles_arr)[::-1]]
    return neighbours_sorted


def trace_edge_from_corner(edges_im, corners_im, p_0):
    cur_path = np.array([p_0])
    cur_vec = np.array([1, 1])
    paths_num = 0
    paths = dict()
    vec_dict = dict()
    visited = {tuple(p_0)}
    relative_neighbours = find_neighbours(edges_im, p_0)  # Relative neighbours.
    neighbours_vecs = relative_neighbours - 1  # Used vectors_to_neighbours(relative_neighbours).
    neighbours = p_0 + neighbours_vecs  # Global neighbours.
    to_visit = sort_neighbours_by_angle(cur_vec, neighbours, neighbours_vecs)
    for i in range(len(neighbours)):
        vec_dict[tuple(neighbours[i])] = neighbours_vecs[i]
    # Running in DFS with vector preferences to build the paths.
    while len(to_visit) != 0:
        # Finding the next pixel in the path.
        cur_pxl = to_visit[-1]
        to_visit = np.delete(to_visit, -1, 0)  # Popping the last element from the numpy array (stack).
        while tuple(cur_pxl) in visited and len(to_visit) != 0:
            cur_pxl = to_visit[-1]
            to_visit = np.delete(to_visit, -1, 0)  # Popping the last element from the numpy array (stack).
        # Adding the new pixel to the data structures.
        if len(cur_path) == 0:
            cur_path = np.append(cur_path, cur_pxl).reshape(1, 2)
        else:
            cur_path = np.append(cur_path, [cur_pxl], axis=0)
        visited.add(tuple(cur_pxl))
        cur_vec = vec_dict[tuple(cur_pxl)]
        # Applying neighbourhood operations on the current pixel.
        relative_neighbours = find_neighbours(edges_im, cur_pxl)  # Relative neighbours.
        neighbours_vecs = relative_neighbours - 1  # Used vectors_to_neighbours(relative_neighbours).
        neighbours = cur_pxl + neighbours_vecs  # Global neighbours.
        neighbours_sorted = sort_neighbours_by_angle(cur_vec, neighbours, neighbours_vecs)  # Global neighbours sorted.
        neighbours_sorted_filtered = np.array([neighbour for neighbour in neighbours_sorted if tuple(neighbour) not in
                                               visited])
        all_neighbours_visited = len(neighbours_sorted_filtered) == 0
        for neighbour in neighbours_sorted_filtered:
            if tuple(neighbour) not in visited:
                to_visit = np.append(to_visit, np.array([neighbour]), axis=0)
        for i in range(len(neighbours)):
            neighbour = tuple(neighbours[i])
            if neighbour not in visited:
                vec_dict[neighbour] = neighbours_vecs[i]
        # If the neighbour is a corner or its neighbours were all visited than it is the end of a path.
        if corners_im[cur_pxl[0], cur_pxl[1]] == 1 or all_neighbours_visited:
            if len(cur_path) > 1:
                paths[paths_num] = cur_path
                paths_num += 1
            cur_path = np.array([])
    return paths


def trace_edges(edges_im, corner_im):
    corners = np.argwhere(corner_im == 1)
    paths = dict()
    paths_num = 0
    for i in range(len(corners)):
        corner = corners[i]
        cur_paths = trace_edge_from_corner(edges_im, corner_im, corner)
        cur_paths_num = len(cur_paths)
        for j in range(cur_paths_num):
            paths[paths_num + j] = cur_paths[j]
        paths_num += cur_paths_num
    return paths


def calculate_bezier_control_points(path):
    path_len = len(path)
    path_len_2 = 2 * path_len
    p_0 = path[0]
    x_y_1 = path[path_len / 3] if path_len % 3 == 0 else (path[path_len // 3] + path[(path_len // 3) + 1]) * 0.5
    x_y_2 = path[path_len_2 / 3] if path_len % 3 == 0 else (path[path_len_2 // 3] + path[(path_len_2 // 3) + 1]) * 0.5
    p_3 = path[-1]
    t_1 = 1 / 3
    t_2 = 2 / 3
    coofficient_1_t1 = 3 * t_1 - 6 * (t_1 ** 2) + 3 * (t_1 ** 3)  # Constant: 4/9.
    coofficient_2_t1 = 3 * (t_1 ** 2) - 3 * (t_1 ** 3)
    coofficient_1_t2 = 3 * t_2 - 6 * (t_2 ** 2) + 3 * (t_2 ** 3)
    coofficient_2_t2 = 3 * (t_2 ** 2) - 3 * (t_2 ** 3)
    x_mat = np.array([[coofficient_1_t1, coofficient_2_t1], [coofficient_1_t2, coofficient_2_t2]])
    x_res_mat = np.array([x_y_1[0], x_y_2[0]])
    y_mat = np.array([[coofficient_1_t1, coofficient_2_t1], [coofficient_1_t2, coofficient_2_t2]])
    y_res_mat = np.array([x_y_1[1], x_y_2[1]])
    x_result = np.linalg.solve(x_mat, x_res_mat)
    y_result = np.linalg.solve(y_mat, y_res_mat)
    p_1 = np.array([x_result[0], y_result[0]])
    p_2 = np.array([x_result[1], y_result[1]])
    return p_0, p_1, p_2, p_3


def recover_bezier_control_points(path, threshold):  # TODO: Not checked and probably wrong. fix recursion return.
    bzr_ctrl_pts = calculate_bezier_control_points(path)
    curve = Rasterizer.bezier_curve_points(bzr_ctrl_pts)
    err = len(np.argwhere(np.linalg.norm(path - curve) != 0))
    bzr_ctrl_pts_dict = dict()
    if err > threshold and len(path > 1):
        bzr_ctrl_pts_dict[0] = recover_bezier_control_points(path[:len(path)//2], threshold)
        bzr_ctrl_pts_dict[1] = recover_bezier_control_points(path[len(path)//2:], threshold)
        return bzr_ctrl_pts_dict
    bzr_ctrl_pts_dict[0] = path
    return bzr_ctrl_pts_dict


def trace_edges_to_bezier(edges_im, corner_im):
    cur_edges_im = edges_im
    cur_curve_im = np.zeros(edges_im.shape)
    neighbors_stack = np.array([])
    bezier_curve_arr = np.array([])
    err = np.max(cur_edges_im - cur_curve_im)


def vectorize_image(im):
    edges_im = detect_edges(im)
    corners_im = detect_corners(edges_im)
    # corners_pairs = pair_corners(edges_im, corners_im)
    # return corners_im
    # return 0.5 * (corners_im + edges_im)
    # p = np.random.randint(1, 11) / 10  # For showreel
    # return p * (corners_im + edges_im)  # For showreel
