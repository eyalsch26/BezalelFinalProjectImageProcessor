import os
import numpy as np
import scipy.signal
import Colourizer
# from scipy import special
from matplotlib import pyplot as plt
from matplotlib import image


GAUSSIAN_KERNEL = 9


def gaussian_kernel(kernel_size):
    """
    Constructs a square symmetric 2D Gaussian kernel according to the given kernel size. The kernel is normalized.
    :param kernel_size: An integer representing the size of the kernel.
    :return: A 2D numpy array of the gaussian kernel. For example, for kernel_size=3 the function returns
    the normalized kernel: [[1,2,1],[2,4,2],[1,2,1]]/16. The entries are of type float64.
    """
    # For one dimensional kernel:
    # if kernel_size > 1:
    #     return np.array(scipy.special.binom(kernel_size - 1, np.arange(kernel_size))/np.power(2, kernel_size - 1), ndmin=2)
    # else:
    #     return np.array([[1]])
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = scipy.signal.convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = scipy.signal.convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def sobel_kernel(direction='x'):
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    if direction == 'y':
        return kernel.T
    return kernel


def laplacian_kernel():
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


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


def detect_edges(image_frame, gaussian_kernel_size):
    # Creating the image and the blur kernel.
    gray_im = Colourizer.rgb_to_gray(image_frame)
    # mask = gaussian_kernel(gaussian_kernel_size)
    # mask_expanded = kernel_padded(mask, gray_im.shape)
    # Extracting the gradient.
    # Extracting the Laplacian.
    # gray_im_fourier = fourier_transform(gray_im)
    # mask_fourier = fourier_transform(mask_expanded)
    # gradient_im_fourier =
    # gaussian_im = convolve2d(gray_im, LAPLACIAN, 'same')
    gradient_im_x = scipy.signal.convolve2d(gray_im, sobel_kernel('x'), 'same')
    gradient_im_x = 255 * gradient_im_x / np.amax(gradient_im_x)
    gradient_im_y = scipy.signal.convolve2d(gray_im, sobel_kernel('y'), 'same')
    gradient_im_y = 255 * gradient_im_y / np.amax(gradient_im_y)
    gradient_im = [gradient_im_x, gradient_im_y]
    # laplacian_im_fourier = gray_im_fourier * mask_expanded
    # laplacian_im = inverse_fourier_transform(laplacian_im_fourier)
    return gradient_im
    # return 255 * gaussian_im / gaussian_im.max()


# ----------------------------------------------- Graveyard Below ------------------------------------------------------
# def fourier_transform(image_frame):
#     image_dft = np.fft.fft2(np.float64(image_frame))
#     image_dft_shifted = np.fft.fftshift(image_dft)
#     # magnitude_image_dft = np.log(np.abs(image_dft_shifted) + 1)  # TODO: Check if log needed.
#     magnitude_image_dft = np.abs(image_dft_shifted)  # TODO: Check if log needed.
#     # Normalizing.
#     magnitude_image_max = np.amax(magnitude_image_dft)
#     magnitude_image_dft = magnitude_image_dft / magnitude_image_max  # To display the image sound add np.uint8(255*( /))
#     return magnitude_image_dft
#
#
#
# def zero_crossing(im):
#     lap_im = laplacian_image(im)
#     zr_crss = np.where(np.diff(np.sign(lap_im)))
#     return zr_crss
#
#
# def sobel_gradient(im):
#     # Creating the image of the kernel.
#     sobel_x = kernel_padder(sobel_kernel('x'), im.shape)
#     sobel_y = kernel_padder(sobel_kernel('y'), im.shape)
#     # Transforming to Fourier domain.
#     gray_im_fourier = dft2D(im)
#     sobel_x_fourier = dft2D(sobel_x)
#     sobel_y_fourier = dft2D(sobel_y)
#     # Multiplying in Fourier domain.
#     frequency_result_x = gray_im_fourier * np.abs(sobel_x_fourier)
#     frequency_result_y = gray_im_fourier * np.abs(sobel_y_fourier)
#     # Back to time domain.
#     im_result_x = idft2D(frequency_result_x)
#     im_result_y = idft2D(frequency_result_y)
#     return im_result_x, im_result_y
#
#
# def quantize_angle_image(im):
#     img = np.abs(im)
#     img[(157.5 < img) & (img <= 22.5)] = 0
#     img[(22.5 < img) & (img <= 67.5)] = 45
#     img[(67.5 < img) & (img <= 112.5)] = 90
#     img[(112.5 < img) & (img <= 157.5)] = 135
#     return img
#
#
# def non_maximum_suppression(image):
#     """
#     Finds local maximas of an image.
#     :param image: A 2D array representing an image.
#     :return: A boolean array with the same shape as the input image, where True indicates local maximum.
#     """
#     # Find local maximas.
#     neighborhood = generate_binary_structure(2, 2)
#     local_max = maximum_filter(image, footprint=neighborhood) == image
#     local_max[image < (image.max()*0.1)] = False
#
#     # Erode areas to single points.
#     lbs, num = label(local_max)
#     centers = center_of_mass(local_max, lbs, np.arange(num)+1)
#     centers = np.stack(centers).round().astype(np.int)
#
#     ret = np.zeros_like(image, dtype=np.bool)
#     ret[centers[:, 0], centers[:, 1]] = True
#
#     return ret
#
#
# def masks_for_nms_canny(nms_size=3):
#     hrz = np.zeros((nms_size, nms_size))
#     hrz[nms_size // 2] = np.ones(nms_size)
#     vtc = hrz.T
#     dce = np.eye(nms_size)
#     acc = dce[::-1]
#     return np.array([hrz, vtc, dce, acc])
#
#
# def separate_gradients(gradient_magnitude, angle_im):
#     sprt_grdt = np.zeros((GRAD_DIRECTIONS_NUM, angle_im.shape[0], angle_im.shape[1]))
#     for i in range(GRAD_DIRECTIONS_NUM):
#         sprt_grdt[i][angle_im == i * QUANTIZE_DEGREE_STEP] = 1
#         sprt_grdt[i] *= gradient_magnitude
#     return sprt_grdt
#
#
# def non_maximum_suppression_canny(gradient_magnitude, im_angle_quantized, nms_size):
#     # Creating the masks according to nms_size.
#     masks = masks_for_nms_canny(nms_size)  # [horizontal, vertical, descent, accent].
#     # Isolating each quantize angle in the gradient magnitude image to separate image.
#     separated_gradients = separate_gradients(gradient_magnitude, im_angle_quantized)  # [horizontal, vertical, descent, accent].
#     # Applying the maxima filter.
#     result = np.zeros(gradient_magnitude.shape)
#     for i in range(GRAD_DIRECTIONS_NUM):
#         result += (maximum_filter(separated_gradients[i], footprint=masks[i]) == gradient_magnitude) * gradient_magnitude
#     return result
#
#
# def canny_edge_detector(im, nms_size=3, k=1.4, t2_cnct=3, gaussian_kernel_size=1):
#     # Generating the gradient magnitude image.
#     blurred_im = blur_image(im, gaussian_kernel_size)
#     im_gradient_sobel = np.gradient(blurred_im)  # sobel_gradient(blurred_im)
#     im_gradient_sobel_x, im_gradient_sobel_y = im_gradient_sobel[0], im_gradient_sobel[1]
#     gradient_magnitude = np.sqrt(im_gradient_sobel_x * im_gradient_sobel_x + im_gradient_sobel_y * im_gradient_sobel_y)
#     # Generating the angle image, converting to degrees and quantizing.
#     im_angle = np.arctan2(im_gradient_sobel_y, im_gradient_sobel_x) * 180.0 / np.pi  # Angles in [-pi, pi].
#     im_angle_quantized = quantize_angle_image(im_angle)
#     # Computing the maxima image using non maximum suppression and normalizing.
#     im_maxima = non_maximum_suppression_canny(gradient_magnitude, im_angle_quantized, nms_size)
#     im_maxima_max = np.max(im_maxima)
#     im_maxima /= im_maxima_max
#     # Computing t1 and t2 and filtering according to them (Hysteresis stage).
#     grd_mean = np.mean(gradient_magnitude / im_maxima_max)
#     sigma = np.std(gradient_magnitude / im_maxima_max)  # Standard deviation.
#     t1 = im_maxima_max * 0.4  # grd_mean + k * sigma
#     t2 = t1 * 0.75  # According to "An improved Canny edge detection algorithm".
#     result = np.zeros(gradient_magnitude.shape)
#     result[im_maxima > t1] = 1  # First filtering.
#     t1_mask = result.copy()
#     t2_mask = np.zeros(gradient_magnitude.shape)  # Second filtering.
#     t2_mask[(im_maxima > t2) & (im_maxima <= t1)] = 1
#     result += maximum_filter(t1_mask, footprint=np.ones((t2_cnct, t2_cnct))) * t2_mask
#     return result
#
#
# def harris_corner_detector_sobel(im, w_size=5, k=0.04, corner_threshold=0.1):
#     # Computing the general form of the M matrix of the whole image.
#     im_gradient = sobel_gradient(im)
#     ix, iy = im_gradient[0], im_gradient[1]
#     ix_square = ix * ix
#     iy_square = iy * iy
#     ixiy = ix * iy
#     # Computing the M matrix entries for each window (the weighted sum of values in the window) by blurring.
#     m_ix_square = ix_square
#     m_iy_square = iy_square
#     m_ixiy = ixiy
#     # Computing the corner response value R in each pixel using the formula: R = Det(M) - k * (Trace(M))^2.
#     m_determinant = m_ix_square * m_iy_square - m_ixiy * m_ixiy
#     m_trace = m_ix_square + m_iy_square
#     r = m_determinant - k * (m_trace * m_trace)  # R is an image(shape==im.shape) where each pixel is a response value.
#     # According to the HUJI exercise.
#     max_m = non_maximum_suppression(r)
#     # result = np.where(max_m == True)
#     # coordinates = np.array(list(zip(result[1], result[0])))
#     # return coordinates
#     return max_m
#
#
# def harris_corner_detector(im, w_size=5, k=0.04, corner_threshold=0.1):
#     # Computing the general form of the M matrix of the whole image.
#     im_gradient = np.gradient(im)
#     ix, iy = im_gradient[0], im_gradient[1]
#     ix_square = ix * ix
#     iy_square = iy * iy
#     ixiy = ix * iy
#     # Computing the M matrix entries for each window (the weighted sum of values in the window) by blurring.
#     m_ix_square = blur_image(ix_square, w_size)
#     m_iy_square = blur_image(iy_square, w_size)
#     m_ixiy = blur_image(ixiy, w_size)
#     # Computing the corner response value R in each pixel using the formula: R = Det(M) - k * (Trace(M))^2.
#     m_determinant = m_ix_square * m_iy_square - m_ixiy * m_ixiy
#     m_trace = m_ix_square + m_iy_square
#     r = m_determinant - k * (m_trace * m_trace)  # R is an image(shape==im.shape) where each pixel is a response value.
#     # According to the HUJI exercise.
#     max_m = non_maximum_suppression(r)
#     # result = np.where(max_m == True)
#     # coordinates = np.array(list(zip(result[1], result[0])))
#     # return coordinates
#     return max_m
#     # # Finding the local maxima in each w*w window. TODO: Might not need to divide to windows.
#     # # Creating a binary image of r's maxima.
#     # r_threshold = corner_threshold * np.max(r)
#     # corners_binary_im = np.zeros(im.shape, np.float64)
#     # corners_binary_im[r >= r_threshold] = 1.0
#     # # corners_binary_im = corners_binary_im * r / np.max(r)
#     # return corners_binary_im
#
# def corner_angles(edges_im):
#     grd_krn = corner_gradient_kernels()
#     crn_ang = np.array([scipy.signal.convolve2d(edges_im, grd_krn[i], mode='same') for i in range(4)]) * edges_im
#     cur = crn_ang[0]
#     hrz = (90 * (1 + cur))[cur != 0]
#     cur = crn_ang[1]
#     vtc = (90 * (2 - cur))[cur != 0]
#     dsc = crn_ang[2]
#     dsc[dsc == -1] = 315
#     dsc[dsc == 1] = 135
#     acc = crn_ang[3]
#     acc[acc == -1] = 45
#     acc[acc == 1] = 225
#     return np.array([hrz, vtc, dsc, acc])
#
#
# def corner_gradient_vectors(corners_grd, edges_im, edges_im_shape):
#     # Creating the basic gradients. Four images to be multiply with the filtered corners.
#     s_e = np.ones((edges_im_shape[0], edges_im_shape[1], 2))
#     n_e = s_e * np.array([1, -1])
#     e = s_e * np.array([1, 0])
#     s = s_e * np.array([0, 1])
#     bsc_grd = np.array([e, s, s_e, n_e])  # Stacking in one array.
#     # Multiplying the basic gradients with the filtered ones to get the orientation of the gradient in each pixel.
#     grd_spr = np.array([bsc_grd[i] * np.expand_dims(corners_grd[i], axis=2) for i in range(GRAD_DIRECTIONS_NUM)])
#     edg_co = np.nonzero(edges_im)
#     for i in range(GRAD_DIRECTIONS_NUM - 1):
#         for j in range(i+1, GRAD_DIRECTIONS_NUM):
#             a = np.dot(grd_spr[i][edg_co], grd_spr[j][edg_co])/(np.abs(grd_spr[i][edg_co]) * np.abs(grd_spr[j][edg_co]))
#
#
# def estimate_ctrl_p_1(edges_im, ctrl_p_0):
#     """
#     Estimates the position of the second Bezier control point. One of the properties of Bezier curves is that the
#     first and last control points tangent to the line formed by the first and second (or third and last respectively)
#     control points. Thus, a rough estimation of the second and third control points will be along the lines described
#     above. Boundaries are checked. A corner or an empty pixel indicates a possible position for the control point.
#     :param edges_im: A numpy array with shape (720, 1280). The entries are of type float64. Represents the edges
#     only (with value of 1 and 0 otherwise). Corners are not included/shown in the edges_im.
#     :param ctrl_p_0: A numpy array with shape (, 2). The entries are of type float64. Represents the initial point to
#     start calculating the Bezier curve from.
#     :return: A tuple of numpy arrays with shape (, 2) where each numpy array with shape (, 2). The entries are of type
#     float64. The first array represents the estimation of the second Bezier control point and the second represents
#     the vector with the direction from the first control point to the second one.
#     """
#     row_s, row_e, column_s, column_e = neighborhood_bounds(ctrl_p_0, edges_im.shape)
#     vec_p_1 = np.argmax(edges_im[row_s:row_e, column_s:column_e])
#     vec_p_1 = np.asarray(np.unravel_index(vec_p_1, (row_e - row_s + 1, column_e - column_s + 1))) - np.array([1, 1])
#     ctrl_p_1 = vec_p_1 + ctrl_p_0
#     in_bounds = check_point_in_bounds(ctrl_p_1 + vec_p_1, edges_im.shape)
#     while in_bounds:
#         ctrl_p_1 += vec_p_1
#         # Checks if the current pixel is an edge and not 0 or corner.
#         pxl_is_edge = edges_im[ctrl_p_1[0]][ctrl_p_1[1]] > 0
#         if not pxl_is_edge:  # If the pixel is a corner or 0.
#             break
#         in_bounds = check_point_in_bounds(ctrl_p_1 + vec_p_1, edges_im.shape)
#     return ctrl_p_1, vec_p_1
#
#
# def estimate_ctrl_p_2(edges_im, next_pxl, p3_p2_vec):
#     ctrl_p_2 = next_pxl + p3_p2_vec
#     in_bounds = check_point_in_bounds(ctrl_p_2 + p3_p2_vec, edges_im.shape)
#     while in_bounds:
#         ctrl_p_2 += p3_p2_vec
#         # Checks if the current pixel is an edge and not 0 or corner.
#         pxl_is_edge = edges_im[ctrl_p_2[0]][ctrl_p_2[1]] > 0
#         if not pxl_is_edge:  # If the pixel is a corner or 0.
#             break
#         in_bounds = check_point_in_bounds(ctrl_p_2 + p3_p2_vec, edges_im.shape)
#     return ctrl_p_2
#
#
# def find_next_pixel(edges_im, pxl):
#     # Checking bounds.
#     row_s, row_e, column_s, column_e = neighborhood_bounds(pxl, edges_im.shape)
#     # Calculating the neighborhood of the current pixel considering the boundaries of the image.
#     neighborhood = edges_im[row_s:row_e, column_s:column_e]
#     # Finding the indices of the neighbors pixels which has value of 1.
#     neighbors = np.argwhere(neighborhood == 1)
#     if len(neighbors) == 0:  # TODO: What if there is no neighbor? Must be referred in the outer scope (calling func).
#         return np.array([-1, -1])
#     # Finds the nearest neighbor by calculating the minimum euclidean distance from the center of the neighborhood.
#     next_pxl_vec = neighbors[np.argmin(np.linalg.norm(neighbors - np.ones((len(neighbors), 2))))] - 1
#     next_pxl = pxl + next_pxl_vec
#     return next_pxl
#
#
# def trim_curve_im(cur_curve_im):
#     """
#     Trims the given image of the curve to provide efficiency in calculation since dealing with smaller images is more efficient.
#     :param cur_curve_im: A numpy array with shape as the shape of the original image frame. The entries are of type
#     np.float64.
#     :return: A tuple with shape (, 2) where the first element is the trimmed image so the frame is tight around the
#     bounds of the curve. The second element is the origin of the trimmed image in respect to the original image (its left
#     most and upper corner).
#     """
#     coord = np.argwhere(cur_curve_im == 1)
#     minima = np.min(coord, axis=0)
#     row_min, column_min = minima[0], minima[1]
#     maxima = np.max(coord, axis=0)
#     row_max, column_max = maxima[0], maxima[1]
#     trimmed = cur_curve_im[row_min:row_max + 1, column_min:column_max + 1]
#     new_origin = [row_min, column_min]
#     return trimmed, new_origin
#
#
# def pad_trimmed_curve_im(trimmed_curve_im, padder_coefficient=3):
#     x_s = trimmed_curve_im.shape[0]
#     y_s = trimmed_curve_im.shape[1]
#     x = padder_coefficient * x_s
#     y = padder_coefficient * y_s
#     padded = np.zeros((x, y))
#     padded[x_s: 2 * x_s, y_s: 2 * y_s] = trimmed_curve_im
#     return padded
#
#
# def detect_corners_0(edges_im):
#     grd_krn = corner_gradient_kernels()
#     crn_grd = np.array([scipy.signal.convolve2d(edges_im, grd_krn[i], mode='same') for i in range(4)]) * edges_im
#     # Creating the basic gradients. Four images to be multiply with the filtered corners.
#     s_e = np.ones((edges_im.shape[0], edges_im.shape[1], 2))
#     n_e = s_e * np.array([1, -1])
#     e = s_e * np.array([1, 0])
#     s = s_e * np.array([0, 1])
#     bsc_grd = np.array([e, s, s_e, n_e])  # Stacking in one array.
#     # Multiplying the basic gradients with the filtered ones to get the orientation of the gradient in each pixel.
#     grd_spr = np.array([bsc_grd[i] * np.expand_dims(crn_grd[i], axis=2) for i in range(GRAD_DIRECTIONS_NUM)])
#     sum_grd = np.sum(grd_spr, axis=0)
#     mgt_grd = np.linalg.norm(sum_grd, axis=2)
#     corners = np.zeros(mgt_grd.shape)
#     corners[(mgt_grd > 1) & (mgt_grd <= 3)] = 1
#     i_corners = scipy.signal.convolve2d(edges_im, one_center_kernel(), mode='same') == 3
#     t_x_crn_flt = scipy.signal.convolve2d(edges_im, t_x_corners_kernel(), mode='same')
#     t_x_corners = np.zeros(edges_im.shape)
#     t_x_corners[(t_x_crn_flt == 7) | (t_x_crn_flt == 9) | (t_x_crn_flt == 41)] = 1
#     y_crn = scipy.signal.convolve2d(edges_im, two_powers_kernel(), mode='same')
#     y_corners = np.zeros(edges_im.shape)
#     y_corners[(y_crn == 549) | (y_crn == 660) | (y_crn == 594) | (y_crn == 585) |
#               (y_crn == 586) | (y_crn == 553) | (y_crn == 676) | (y_crn == 658)] = 1
#     bolt_crn = np.zeros(edges_im.shape)
#     bolt_crn[(y_crn == 556) | (y_crn == 688) | (y_crn == 673) | (y_crn == 618) |
#              (y_crn == 706) | (y_crn == 523) | (y_crn == 538) | (y_crn == 646)] = 1
#     # c_corners = np.zeros(edges_im.shape)
#     # c_corners[(y_crn == 517) | (y_crn == 532) | (y_crn == 592) | (y_crn == 577)] = 1
#     corners += i_corners + t_x_corners + y_corners - bolt_crn
#     return corners
#     # l_corners = np.ones(edges_im.shape)[(tmp == 522) | (tmp == 552) | (tmp == 672) | (tmp == 642)]
#     # r_corners = np.ones(edges_im.shape)[(tmp == 526) | (tmp == 568) | (tmp == 736) | (tmp == 643)] - c_corners - l_corners
#     # x_corners = np.ones(edges_im.shape)[(tmp == 682) | (tmp == 597)]
#
#
# def convert_ctrl_pts(ctrl_p_0, ctrl_p_1, ctrl_p_2, ctrl_p_3, padded_origin):
#     c_p_0_t = ctrl_p_0 - padded_origin
#     c_p_1_t = ctrl_p_1 - padded_origin
#     c_p_2_t = ctrl_p_2 - padded_origin
#     c_p_3_t = ctrl_p_3 - padded_origin
#     return c_p_0_t, c_p_1_t, c_p_2_t, c_p_3_t
#
#
# def find_bezier_ctrl_points(ctrl_p_0, ctrl_p_1, ctrl_p_2, ctrl_p_3, curve_im, search_step=0.5):
#     cur_ctrl_p_1, cur_ctrl_p_2 = ctrl_p_1, ctrl_p_2
#     bezier_control_points = np.array([ctrl_p_0, cur_ctrl_p_1, cur_ctrl_p_2, ctrl_p_3])
#     cur_result = Rasterizer.bezier_curve_rasterizer(bezier_control_points, canvas_shape=curve_im.shape)
#     err = curve_im - cur_result
#     r_srch_rds = int(curve_im.shape[0] / 3) - 1
#     c_srch_rds = int(curve_im.shape[1] / 3) - 1
#     r_intervals = int((search_step ** (-1)) * r_srch_rds + 1)
#     c_intervals = int((search_step ** (-1)) * c_srch_rds + 1)
#     if np.min(err) < 0:  # TODO: Allow out of bounds - find a way for control points outside the frame.
#         for p_1_r_ad in np.linspace(-r_srch_rds, r_srch_rds, r_intervals):
#             for p_1_c_ad in np.linspace(-c_srch_rds, c_srch_rds, c_intervals):
#                 for p_2_r_ad in np.linspace(-r_srch_rds, r_srch_rds, r_intervals):
#                     for p_2_c_ad in np.linspace(-c_srch_rds, c_srch_rds, c_intervals):
#                         cur_ctrl_p_1 = ctrl_p_1 + np.array([p_1_r_ad, p_1_c_ad])
#                         cur_ctrl_p_2 = ctrl_p_2 + np.array([p_2_r_ad, p_2_c_ad])
#                         bezier_control_points = np.array([ctrl_p_0, cur_ctrl_p_1, cur_ctrl_p_2, ctrl_p_3])
#                         cur_result = Rasterizer.bezier_curve_rasterizer(bezier_control_points,
#                                                                         canvas_shape=curve_im.shape)
#                         err = curve_im - cur_result
#                         if np.min(err) == 0:
#                             return ctrl_p_0, cur_ctrl_p_1, cur_ctrl_p_2, ctrl_p_3
#     if np.min(err) < 0:
#         return ctrl_p_0, ctrl_p_1, ctrl_p_2, ctrl_p_3
#     return ctrl_p_0, cur_ctrl_p_1, cur_ctrl_p_2, ctrl_p_3
#
#
# def trace_edge_to_bezier(edges_im, corner_im, ctrl_p_0):
#     """
#
#     :param edges_im: A numpy array with shape (720, 1280). The entries are of type float64. Represents the edges and
#     corners (with value of 1 and 0 otherwise).
#     :param corner_im: A numpy array with shape (720, 1280). The entries are of type float64. Represents the corners
#     only (with value of 1 and 0 otherwise). Edges are not included.
#     :param ctrl_p_0: A numpy array with shape (, 2). The entries are of type float64. Represents the initial point to
#     start calculating the Bezier curve from.
#     :return:
#     """
#     cur_edges_im = edges_im
#     cur_curve_im = np.zeros(edges_im.shape)
#     curve_len = 0
#     # Estimating the location of the second bezier control point and a vector from the first to second control points.
#     ctrl_p_1, p0_p1_vec = estimate_ctrl_p_1(edges_im - corner_im, ctrl_p_0)
#     cur_pxl = ctrl_p_0
#     # Building the current curve image until the first control point.
#     while cur_pxl != ctrl_p_1:
#         cur_curve_im[cur_pxl[0]][cur_pxl[1]] = 1
#         cur_pxl += p0_p1_vec
#         curve_len += 1
#     cur_pxl = ctrl_p_1 - p0_p1_vec  # TODO: Consider at least 6 pixels.
#     cur_edges_im -= cur_curve_im
#     next_pxl = find_next_pixel(cur_edges_im, cur_pxl)
#     while curve_len < 5:
#         cur_pxl = next_pxl
#         cur_curve_im[cur_pxl[0]][cur_pxl[1]] = 1
#         cur_edges_im -= cur_curve_im
#         next_pxl = find_next_pixel(cur_edges_im, cur_pxl)
#         curve_len += 1
#     p3_p2_vec = cur_pxl - next_pxl
#     cur_curve_im[next_pxl[0]][next_pxl[1]] = 1
#     ctrl_p_2 = estimate_ctrl_p_2(edges_im, next_pxl, p3_p2_vec)
#     # Trimming the relevant window where the curve is included from the entire image.
#     cur_curve_im_trim, trimmed_origin = trim_curve_im(cur_curve_im)
#     # Padding the trimmed curve image to allow control points outside the frame of the original image.
#     cur_curve_im_trim_pad = pad_trimmed_curve_im(cur_curve_im_trim)
#     padded_origin = trimmed_origin - np.array([cur_curve_im_trim.shape[0], cur_curve_im_trim.shape[1]])
#     c_p_0_t, c_p_1_t, c_p_2_t, c_p_3_t = convert_ctrl_pts(ctrl_p_0, ctrl_p_1, ctrl_p_2, next_pxl, padded_origin)
#     cur_bezier_ctrl_pts = find_bezier_ctrl_points(c_p_0_t, c_p_1_t, c_p_2_t, c_p_3_t, cur_curve_im_trim_pad)
#
#
# def trace_edges_to_bezier0(edges_im, corner_im):
#     bezier_control_points = np.zeros((1, 2, 4))
#     cur_corners_im = corner_im
#     cur_edges_im = 2 * edges_im - corner_im  # Edges=2. Corners=1. Empty=0.
#     while np.max(cur_corners_im) > 0:
#         # Defining the first bezier point.
#         ctrl_p_0 = np.asarray(np.unravel_index(np.argmax(cur_corners_im), corner_im.shape))
#         # Computing the second bezier point. Might be adjusted later.
#         vec_p_1 = np.asarray(np.unravel_index(np.argmax(cur_edges_im[ctrl_p_0[0]-1:ctrl_p_0[0]+1,
#                                                         ctrl_p_0[1]-1:ctrl_p_0[1]+1]), (3, 3))) - np.array([1, 1])
#         ctrl_p_1 = 2 * vec_p_1 + ctrl_p_0  # TODO: Index out of bounds.
#         while cur_edges_im[ctrl_p_1[0]][ctrl_p_1[1]] > 0:  # TODO: Index out of bounds.
#             ctrl_p_1 += vec_p_1
#         turns_num = 1
#
#         corners_num = np.sum(cur_corners_im)
#         cur_curve = np.zeros((3, 3))
#         # Updating the current corner image by removing isolated corners.
#
#         cur_corners_im *= remove_isolated_pixels(cur_edges_im)
#     return bezier_control_points
#
#
# def find_connected_corners(edges_im, corners_im, corner):
#     connected_corners = set()
#     visited = set(tuple(corner))
#     to_visit = find_neighbours(edges_im, corner)
#     while len(to_visit) != 0:
#         nxt_pxl = to_visit[-1]
#         np.delete(to_visit, -1, 0)  # Popping the last element from the numpy array (stack).
#         visited.add(tuple(nxt_pxl))
#         # If the neighbour is a corner than it is added to the set and we continue to check the other neighbours.
#         if corners_im[nxt_pxl[0], nxt_pxl[1]] == 1:
#             connected_corners.add(nxt_pxl)
#             continue
#         neighbours = find_neighbours(edges_im, nxt_pxl)
#         for neighbour in neighbours:
#             if tuple(neighbour) not in visited:
#                 np.append(to_visit, neighbour, axis=0)
#
#
# def pair_corners(edges_im, corners_im):
#     corner_pairs = dict()
#     corners = np.argwhere(corners_im == 1)
#     for corner in corners:
#         connected_corners = find_connected_corners(edges_im, corners_im, corner)
#         # Checks for duplicates i.e. the current corner and its neighbor are already in the dictionary as value and key.
#         for connected_corner_idx in range(len(connected_corners)):
#             if corner_pairs[tuple(connected_corners[connected_corner_idx])] == tuple(corner):
#                 connected_corners = np.delete(connected_corners, connected_corner_idx, 0)
#         corner_pairs[tuple(corner)] = connected_corners
#     return corner_pairs

