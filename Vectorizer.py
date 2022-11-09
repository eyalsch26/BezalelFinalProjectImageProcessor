import numpy as np
import scipy.signal
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import Colourizer


GAUSSIAN_KERNEL = 9
HARRIS_W = 5


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
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64) / 4.0
    if direction == 'y':
        return kernel.T
    return kernel


def laplacian_kernel():
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64) / 4.0


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


def sobel_gradient(im):
    # Creating the image of the kernel.
    sobel_x = kernel_padder(sobel_kernel('x'), im.shape)
    sobel_y = kernel_padder(sobel_kernel('y'), im.shape)
    # Transforming to Fourier domain.
    gray_im_fourier = dft2D(im)
    sobel_x_fourier = dft2D(sobel_x)
    sobel_y_fourier = dft2D(sobel_y)
    # Multiplying in Fourier domain.
    frequency_result_x = gray_im_fourier * np.abs(sobel_x_fourier)
    frequency_result_y = gray_im_fourier * np.abs(sobel_y_fourier)
    # Back to time domain.
    im_result_x = idft2D(frequency_result_x)
    im_result_y = idft2D(frequency_result_y)
    return im_result_x, im_result_y


def quantize_angle_image(im):
    im[im <= 22.5] = 0
    im[22.5 < im <= 67.5] = 45
    im[67.5 < im <= 112.5] = 90
    # im[112.5 < im <= ] = 135
    im[im < -135] = 135


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max()*0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num)+1)
    centers = np.stack(centers).round().astype(np.int)

    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


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


def canny_edge_detector(im, gaussian_kernel_size, t_1, t_2):
    im_gradient_sobel = sobel_gradient(im)
    im_angle = np.abs(np.arctan2(im_gradient_sobel[1], im_gradient_sobel[0]) * 180.0 / np.pi)  # Angles in [-pi, pi].
    im_angle_quantized = quantize_angle_image(im_angle)


def harris_corner_detector_sobel(im, w_size=5, k=0.04, corner_threshold=0.1):
    # Computing the general form of the M matrix of the whole image.
    im_gradient = sobel_gradient(im)
    ix, iy = im_gradient[0], im_gradient[1]
    ix_square = ix * ix
    iy_square = iy * iy
    ixiy = ix * iy
    # Computing the M matrix entries for each window (the weighted sum of values in the window) by blurring.
    m_ix_square = ix_square
    m_iy_square = iy_square
    m_ixiy = ixiy
    # Computing the corner response value R in each pixel using the formula: R = Det(M) - k * (Trace(M))^2.
    m_determinant = m_ix_square * m_iy_square - m_ixiy * m_ixiy
    m_trace = m_ix_square + m_iy_square
    r = m_determinant - k * (m_trace * m_trace)  # R is an image(shape==im.shape) where each pixel is a response value.
    # According to the HUJI exercise.
    max_m = non_maximum_suppression(r)
    # result = np.where(max_m == True)
    # coordinates = np.array(list(zip(result[1], result[0])))
    # return coordinates
    return max_m


def harris_corner_detector(im, w_size=5, k=0.04, corner_threshold=0.1):
    # Computing the general form of the M matrix of the whole image.
    im_gradient = np.gradient(im)
    ix, iy = im_gradient[0], im_gradient[1]
    ix_square = ix * ix
    iy_square = iy * iy
    ixiy = ix * iy
    # Computing the M matrix entries for each window (the weighted sum of values in the window) by blurring.
    m_ix_square = blur_image(ix_square, w_size)
    m_iy_square = blur_image(iy_square, w_size)
    m_ixiy = blur_image(ixiy, w_size)
    # Computing the corner response value R in each pixel using the formula: R = Det(M) - k * (Trace(M))^2.
    m_determinant = m_ix_square * m_iy_square - m_ixiy * m_ixiy
    m_trace = m_ix_square + m_iy_square
    r = m_determinant - k * (m_trace * m_trace)  # R is an image(shape==im.shape) where each pixel is a response value.
    # According to the HUJI exercise.
    max_m = non_maximum_suppression(r)
    # result = np.where(max_m == True)
    # coordinates = np.array(list(zip(result[1], result[0])))
    # return coordinates
    return max_m
    # # Finding the local maxima in each w*w window. TODO: Might not need to divide to windows.
    # # Creating a binary image of r's maxima.
    # r_threshold = corner_threshold * np.max(r)
    # corners_binary_im = np.zeros(im.shape, np.float64)
    # corners_binary_im[r >= r_threshold] = 1.0
    # # corners_binary_im = corners_binary_im * r / np.max(r)
    # return corners_binary_im
