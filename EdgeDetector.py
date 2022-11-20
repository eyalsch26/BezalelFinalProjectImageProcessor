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
