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


# -------------------------------------------------- Image Tracing -----------------------------------------------------

def gaussian_kernel(kernel_size):
    """
    Constructs a square symmetric 2D Gaussian kernel according to the given kernel size. The kernel is normalized.
    :param kernel_size: An integer representing the size of the kernel.
    :return: A 2D numpy array of the gaussian kernel where the entries are of type float64.
    For example, for kernel_size=3 the function returns the normalized kernel: [[1,2,1],
                                                                                [2,4,2],
                                                                                [1,2,1]] / 16.
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
    """
    Expands the given kernel to the shape of the original image in order to preform a blur using dft.
    :param kernel: A numpy array with shape (x, x) where x is an odd number (all possible shapes of a square kernel).
    It's elements are integers.
    :param im_shape: A tuple with shape (, 2) with dtype of integers. For example (1080, 1920).
    :return: A numpy array with shape im_shape where the elements are of type np.float64. The given gernel is located
    in the center of the returned image.
    """
    expanded_kernel = np.zeros(im_shape, np.float64)
    top_left = int(np.round((im_shape[0] - kernel.shape[0]) * 0.5))
    top_right = top_left + kernel.shape[0]
    bottom_left = int(np.round((im_shape[1] - kernel.shape[1]) * 0.5))
    bottom_right = bottom_left + kernel.shape[1]
    expanded_kernel[top_left:top_right, bottom_left:bottom_right] = kernel
    return expanded_kernel


def dft2D(image_frame):
    """
    Preforms discrete fourier transform on the given image.
    :param image_frame: A numpy array with dtype np.float64. The entries are in the range of [0, 1].
    :return: A numpy array with dtype np.float64. This is the image in the fourier domain.
    """
    image_dft = np.fft.fft2(np.float64(image_frame))
    image_dft_shifted = np.fft.fftshift(image_dft)
    return image_dft_shifted


def idft2D(fourier_image):
    """
    Preforms inverse discrete fourier transform on the given image.
    :param fourier_image: A numpy array with dtype np.float64. This is the original image in the fourier domain.
    :return: A numpy array with dtype np.float64. This is the image in the time domain.
    """
    fourier_image_shifted_back = np.fft.ifftshift(fourier_image)
    return np.real(np.fft.ifft2(fourier_image_shifted_back))


def blur_image(im, gaussian_kernel_size):
    """
    Blurs the given image by creating a blur kernel according to the given kernel size, padding the kernel to the
    shape of the given image, transforming both to the fourier domain, applying matrix multiplication element wise
    :param im: A numpy array with dtype np.float64. The entries are in the range of [0, 1].
    :param gaussian_kernel_size: An odd integer representing the size of the gaussian kernel. For understanding
    using example see the documentation of the gaussian_kernel() function.
    :return: A numpy array with shape im.shape where the entries are of dtype np.float64. The returned image is the
    blurred original image.
    """
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
    """
    Calculates the Laplacian of the given image. A reminder: The Laplacian is the divergence of the gradient,
    means it is the sum of the second derivatives of the original image:
    Laplacian = second_derivative_x(im) + second_derivative_y(im).
    :param im: A numpy array with dtype np.float64. The entries are in the range of [0, 1]. This is the original
    image to compute the Laplacian upon.
    :return: A numpy array with shape im.shape and with dtype np.float64. This is the Laplacian image.
    """
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
    """
    removes pixels which have no neighbours from first order in the given image.
    :param im: A numpy array with dtype np.float64. The entries are in the range of [0, 1].
    :return: A numpy array with shape im.shape and with dtype np.float64. In this image a box in the form of:
    [[0, 0, 0],
     [0, x, 0],
     [0, 0, 0]] where x is a number of dtype np.float64 in the range of [0, 1] cannot be found.
    """
    # Creating the image of the kernel.
    mask = np.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]], dtype=np.float64)
    # Convolving to remove the isolated pixels.
    im_result = im - (scipy.signal.convolve2d(im, mask, mode='same') == 1)
    return im_result


def thin_edges(im):
    """
    Diminishes the edges to a width of one pixel. Only one iteration is applied so some leftovers might be remained.
    :param im: A numpy array with dtype np.float64.
    :return: A numpy array with shape im.shape and with dtype np.float64.
    """
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
    """
    Removes the isolated pixels and thins the edges to one pixel width in the image.
    :param im: A numpy array with dtype np.float64.
    :return: A numpy array with shape im.shape and with dtype np.float64.
    """
    clean_im = thin_edges(im)
    cleaner_im = remove_isolated_pixels(clean_im)
    return cleaner_im


def detect_edges(im, t1_co=0.975, t2_co=0.995):
    """
    Finds the edges in the image based on the canny detection algorithm. The low filter is set to 0.975 and the high
    filter is set to 0.995 as default.
    :param im: A numpy array with dtype np.float64.
    :param t1_co: A floating point number representing the low pass filter so that pixels with values lower than
    t1_co will be removed from the final image and will not represent an edge.
    :param t2_co: A floating point number representing the high pass filter so that pixels with values higher than
    t2_co will be removed from the final image and will not represent an edge.
    :return: A numpy array with shape im.shape and with dtype np.float64. In this image only the edges remaind.
    """
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
    """
    Builds four matrices to find the gradient along the four axes: horizontal, vertical, descending diagonal (from
    left to right) and ascending diagonal (from left to right).
    :return: A numpy array with shape (4, 3, 3) with dtype integer.
    """
    hrz = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    vtc = hrz.T
    dsc = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]])
    acc = dsc[::-1]
    return np.array([hrz, vtc, dsc, acc])


def detect_corners(edges_im):
    """
    Finds the coordinates in the given image where there are corners in the following forms (42 forms overall):
    1. 'i_corners': 0 1 0   0 0 1   0 0 0   0 0 0   0 0 0   0 0 0   0 0 0   1 0 0
                    0 1 0   0 1 0   0 1 1   0 1 0   0 1 0   0 1 0   1 1 0   0 1 0
                    0 0 0   0 0 0   0 0 0   0 0 1   0 1 0   1 0 0   0 0 0   0 0 0

    2. 'c_corners': 1 0 1   0 0 1   0 0 0   1 0 0
                    0 1 0   0 1 0   0 1 0   0 1 0
                    0 0 0   0 0 1   1 0 1   1 0 0

    3. 'l_corners': 0 1 0   0 0 0   0 0 0   0 1 0
                    0 1 1   0 1 1   1 1 0   1 1 0
                    0 0 0   0 1 0   0 1 0   0 0 0

    4. 'r_corners': 0 1 1   0 0 0   0 0 0   1 0 0   1 1 0   0 0 0   0 0 0   0 0 1
                    0 1 0   0 1 1   0 1 0   1 1 0   0 1 0   1 1 0   0 1 0   0 1 1
                    0 0 0   0 0 1   1 1 0   0 0 0   0 0 0   1 0 0   0 1 1   0 0 0

    5. 't_corners': 0 1 0   0 0 0   0 1 0   0 1 0   1 0 1   1 0 0   0 0 1   1 0 1
                    1 1 1   1 1 1   1 1 0   0 1 1   0 1 0   0 1 0   0 1 0   0 1 0
                    0 0 0   0 1 0   0 1 0   0 1 0   0 0 1   1 0 1   1 0 1   1 0 0

    6. 'x_corners': 0 1 0   1 0 1
                    1 1 1   0 1 0
                    0 1 0   1 0 1

    7. 'y_corners': 0 1 0   1 0 0   0 0 1   0 1 0   1 0 1   1 0 0   0 1 0   0 0 1
                    0 1 1   0 1 1   1 1 0   1 1 0   0 1 0   0 1 1   0 1 0   1 1 0
                    1 0 0   0 1 0   0 1 0   0 0 1   0 1 0   1 0 0   1 0 1   0 0 1
    :param edges_im: A numpy array with dtype np.float64.
    :return: A numpy array with shape edges_im.shape and with dtype np.float64 in the range [0, 1]. In this image each
    entry with value of 1 represents a corner in the image.
    """
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
    """
    Given a coordinate of a pixel and the shape of the frame, checks if a box of 3x3 around the given pixel
    coordinates fits in the shape of the frame to ensure indices in bounds (to avoid "index out of bounds exception").
    :param pxl: A numpy array with shape (, 2) which represents the pixel to check the neighbourhood bounds around.
    :param shape: A tuple with shape (, 2). For example (1080, 1920).
    :param r_d: An integer represents the row radius from the given pixel (how many rows to check before and how many
    rows to check after the given x coordinate of the given pixel). For example, if pixel=[4,3] and r_d=2,
    the function checks if indices 4-2=2 and 4+2=6 are in bounds respective to the shape given.
    :param c_d: An integer represents the column radius from the given pixel as described above in r_d.
    :return: A tuple with shape (, 4) with entries of type integers representing the indices of the neighbourhood of
    the given pixel.
    """
    row_s = pxl[0] - r_d if pxl[0] > 0 else 0
    row_e = pxl[0] + r_d + 1 if pxl[0] < shape[0] - r_d else shape[0]
    column_s = pxl[1] - c_d if pxl[1] > 0 else 0
    column_e = pxl[1] + c_d + 1 if pxl[1] < shape[1] - c_d else shape[1]
    return row_s, row_e, column_s, column_e


def find_neighbours(edges_im, pxl):
    """
    Detects the pixels with value of 1 in a radius of one pixel from the given pixel. These pixels are called
    neighbours and the group of these pixels is called a neighbourhood. For example, in the following case:
    0 1 0 0 0 0 0
    0 0 1 0 0 1 1
    0 0 0 1 0 0 1
    0 0 0 1 0 0 0
    the neighbourhood of the pixel in the coordinate [2,3] to be returned will be: [[0,0], [2,1]]. Notice that the
    coordinates are relative to the 3x3 box around the given pixel.
    :param edges_im: A numpy array with dtype np.float64. The image of the current edges.
    :param pxl: A numpy array with shape (,2) and dtype np.float64. The coordinate of the pixel to detect the
    neighbours around.
    :return: A numpy array with dtype np.float64 and shape (1, x, 2) where 0<x<8 is the number of neighbours.
    """
    # Checking bounds.
    row_s, row_e, column_s, column_e = neighborhood_bounds(pxl, edges_im.shape)
    # Calculating the neighborhood of the current pixel considering the boundaries of the image.
    neighborhood = np.copy(edges_im[row_s:row_e, column_s:column_e])
    # Deleting the origin pixel (a pixel is not a neighbour of itself).
    neighborhood[1][1] = 0
    # Finding the indices of the neighbors pixels which has value of 1.
    neighbours = np.argwhere(neighborhood == 1)
    return neighbours


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
            cur_path = np.append(cur_path, cur_pxl).reshape(1, 2)  # TODO: Check the reason for the reshape.
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


def trace_edges_to_paths(edges_im, corner_im):
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


def partition_indices(path):
    path_len = len(path)
    path_len_2 = 2 * path_len
    partition_1 = path_len // 3
    partition_2 = path_len_2 // 3
    if path_len % 3 != 0 and path_len > 2:
        x_y_1 = (path[partition_1] + path[partition_1 + 1]) * 0.5
        x_y_2 = (path[partition_2] + path[partition_2 + 1]) * 0.5
    else:
        x_y_1 = (path[partition_1])
        x_y_2 = (path[partition_2])
    return x_y_1, x_y_2


def calculate_bezier_control_points(path):
    # Setting the four points on the curve.
    p_0 = path[0]
    x_y_1, x_y_2 = partition_indices(path)
    p_3 = path[-1]
    # Calculating constants.
    t_1 = 1 / 3
    t_2 = 2 / 3
    coefficient_0_t1 = 1 - 3 * t_1 + 3 * (t_1 ** 2) - t_1**3
    coefficient_1_t1 = 3 * t_1 - 6 * (t_1 ** 2) + 3 * (t_1 ** 3)  # Constant: 4/9.
    coefficient_2_t1 = 3 * (t_1 ** 2) - 3 * (t_1 ** 3)
    coefficient_3_t1 = t_1 ** 3
    coefficient_0_t2 = 1 - 3 * t_2 + 3 * (t_2 ** 2) - t_2**3
    coefficient_1_t2 = 3 * t_2 - 6 * (t_2 ** 2) + 3 * (t_2 ** 3)
    coefficient_2_t2 = 3 * (t_2 ** 2) - 3 * (t_2 ** 3)
    coefficient_3_t2 = t_2 ** 3
    # Calculating the linear equation system's coefficients.
    # The left hand side of the linear equation system.
    x_mat = np.array([[coefficient_1_t1, coefficient_2_t1], [coefficient_1_t2, coefficient_2_t2]])
    y_mat = np.array([[coefficient_1_t1, coefficient_2_t1], [coefficient_1_t2, coefficient_2_t2]])
    # The right hand side of the linear equation system.
    x_res_0 = x_y_1[0] - (p_0[0] * coefficient_0_t1) - (p_3[0] * coefficient_3_t1)
    x_res_1 = x_y_2[0] - (p_0[0] * coefficient_0_t2) - (p_3[0] * coefficient_3_t2)
    y_res_0 = x_y_1[1] - (p_0[1] * coefficient_0_t1) - (p_3[1] * coefficient_3_t1)
    y_res_1 = x_y_2[1] - (p_0[1] * coefficient_0_t2) - (p_3[1] * coefficient_3_t2)
    x_res_mat = np.array([x_res_0, x_res_1])
    y_res_mat = np.array([y_res_0, y_res_1])
    # Solving the linear equation system to get the two missing control points.
    x_result = np.linalg.solve(x_mat, x_res_mat)
    y_result = np.linalg.solve(y_mat, y_res_mat)
    p_1 = np.array([x_result[0], y_result[0]])
    p_2 = np.array([x_result[1], y_result[1]])
    return np.array([p_0, p_1, p_2, p_3])


def calculate_path_curve_error(path, curve):
    max_len = np.max((len(path), len(curve)))
    max_dim = np.max((int(np.max(path)), int(np.max(curve)))) + 1
    path_im = np.zeros((max_dim, max_dim))
    curve_im = np.zeros((max_dim, max_dim))
    path_indices = path.T[0] * 2 + path.T[1]
    curve_indices = curve.T[0] * 2 + curve.T[1]
    np.put(path_im, path_indices.astype(np.int32), 1.0)
    np.put(curve_im, curve_indices.astype(np.int32), 1.0)
    error_im = path_im - curve_im
    error = len(np.argwhere(error_im != 0)) / max_len
    return error


def recover_bezier_control_points(path, threshold=0.25):
    bzr_ctrl_pts = calculate_bezier_control_points(path)
    curve = Rasterizer.bezier_curve_points(bzr_ctrl_pts)
    err = calculate_path_curve_error(path, curve)
    bzr_ctrl_pts_dict = dict()
    if err > threshold and len(path > 1):
        bzr_ctrl_pts_dict_0 = recover_bezier_control_points(path[:1 + len(path)//2], threshold)
        bzr_ctrl_pts_dict_1 = recover_bezier_control_points(path[len(path)//2:], threshold)
        bzr_curve_num_0 = len(bzr_ctrl_pts_dict_0)
        bzr_curve_num_1 = len(bzr_ctrl_pts_dict_1)
        for i in range(bzr_curve_num_0):
            bzr_ctrl_pts_dict[i] = bzr_ctrl_pts_dict_0[i]
        for j in range(bzr_curve_num_1):
            bzr_ctrl_pts_dict[bzr_curve_num_0 + j] = bzr_ctrl_pts_dict_1[j]
        return bzr_ctrl_pts_dict
    bzr_ctrl_pts_dict[0] = bzr_ctrl_pts
    return bzr_ctrl_pts_dict


def trace_edges_to_bezier(edges_im, corner_im):
    paths_dict = trace_edges_to_paths(edges_im, corner_im)
    bzr_ctrl_pts_dict = dict()
    curves_connectivity_arr = np.array([])
    paths_num = len(paths_dict)
    curves_num = 0
    for p in range(paths_num):
        cur_bzr_ctrl_pts_dict = recover_bezier_control_points(paths_dict[p])
        cur_curves_num = len(cur_bzr_ctrl_pts_dict)
        for c in range(cur_curves_num):
            bzr_ctrl_pts_dict[curves_num + c] = cur_bzr_ctrl_pts_dict[c]
        curves_num += cur_curves_num
        curves_connectivity_arr = np.append(curves_connectivity_arr, cur_curves_num)
    return bzr_ctrl_pts_dict, curves_connectivity_arr


def vectorize_image(im):
    edges_im = detect_edges(im)
    corners_im = detect_corners(edges_im)
    bzr_ctrl_pts_dict, connectivity_arr = trace_edges_to_bezier(edges_im, corners_im)
    bzr_ctrl_pts_arr = np.array([bzr_ctrl_pts_dict[i] for i in range(len(bzr_ctrl_pts_dict))])
    return bzr_ctrl_pts_arr
    # p = np.random.randint(1, 11) / 10  # For showreel
    # return p * (corners_im + edges_im)  # For showreel


# ----------------------------------------------- Vector Manipulation --------------------------------------------------

def perpendicular_vec(vec, norm='same'):
    p_vec = np.array([-vec[1], vec[0]])
    if norm == 'one':
        p_vec /= np.linalg.norm(p_vec)
    return p_vec


def displace_bezier_control_points(bezier_control_points):
    # Setting local variables.
    p_0 = bezier_control_points[0]
    p_1 = bezier_control_points[1]
    p_2 = bezier_control_points[2]
    p_3 = bezier_control_points[3]
    # Calculating the three vectors created by the four points (as described in the documentation and the notebook).
    a = p_1 - p_0
    b = p_2 - p_1
    c = p_2 - p_3
    d = p_3 - p_0
    e = perpendicular_vec(d)  # The perpendicular vector to d. (Dot product == 0).
    # Generating random coefficients to multiply the vectors by.
    rndm_pos_coefs = np.random.randint(1, 6, 2) * 0.4  # Range [1, 2] in steps of 0.4.
    rndm_neg_pos_coefs = np.random.randint(-5, 6, 4) * 0.4  # Range [-2, 2] in steps of 0.4.
    p_0_coef_d, p_3_coef_d = rndm_pos_coefs[0], rndm_pos_coefs[1]
    p_1_coef, p_2_coef = rndm_neg_pos_coefs[0], rndm_neg_pos_coefs[1]
    p_0_coef_e, p_3_coef_e = rndm_neg_pos_coefs[2], rndm_neg_pos_coefs[3]
    # Calculating the vectors.
    p_0_vec = - p_0_coef_d * d + p_0_coef_e * e
    p_1_vec = p_1_coef * (a - b)
    p_2_vec = p_2_coef * (b + c)
    p_3_vec = - p_3_coef_d * d + p_3_coef_e * e  # Was: - (p_3_coef * c - b)
    # Calculating the new control points.
    p_0_new = p_0 + p_0_vec
    p_1_new = p_1 + p_1_vec
    p_2_new = p_2 + p_2_vec
    p_3_new = p_3 + p_3_vec
    return np.array([p_0_new, p_1_new, p_2_new, p_3_new])  # TODO: Check for index out of bounds.


def displace_bezier_curves(bezier_control_points_arr):
    new_bzr_ctrl_pts_arr = np.array([])
    for bzr_ctrl_pts in bezier_control_points_arr:
        new_bzr_ctrl_pts = displace_bezier_control_points(bzr_ctrl_pts)
        np.append(new_bzr_ctrl_pts_arr, [new_bzr_ctrl_pts], axis=0)
    return new_bzr_ctrl_pts_arr
