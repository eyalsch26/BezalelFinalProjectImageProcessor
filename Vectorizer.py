import numpy as np
import scipy.signal
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import Colourizer


GAUSSIAN_KERNEL = 9
HARRIS_W = 5
GRAD_DIRECTIONS_NUM = 4
QUANTIZE_DEGREE_STEP = 45


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
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)  # / 4.0
    if direction == 'y':
        return kernel.T
    return kernel


def laplacian_kernel():
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def two_powers_kernel():
    return np.array([[1, 2, 4], [128, 512, 8], [64, 32, 16]], dtype=np.float64)


def one_center_kernel():
    return np.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]], dtype=np.float64)


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
    tep = scipy.signal.convolve2d(im, two_powers_kernel(), mode='same')
    im[(tep == 519) | (tep == 540) | (tep == 624) | (tep == 705)] = 0  # Removing spikes. E.g.: [0,0,0],[0,1,0],[1,1,1]
    im[(tep == 574) | (tep == 760)] = 0  # Removing double edges. E.g.: [0,0,0],[1,1,1],[1,1,1]
    return im


def clean_undesired_pixels(im):
    clean_im = remove_isolated_pixels(im)
    cleaner_im = thin_edges(clean_im)
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


def detect_corners(edges_im):
    i_corners = scipy.signal.convolve2d(edges_im, one_center_kernel(), mode='same') == 3
    tmp = scipy.signal.convolve2d(edges_im, two_powers_kernel(), mode='same')
    c_corners = np.ones(edges_im.shape)[(tmp == 517) | (tmp == 532) | (tmp == 592) | (tmp == 577)]
    l_corners = np.ones(edges_im.shape)[(tmp == 522) | (tmp == 552) | (tmp == 672) | (tmp == 642)]
    r_corners = np.ones(edges_im.shape)[(tmp == 526) | (tmp == 568) | (tmp == 736) | (tmp == 643)] - c_corners - l_corners
    x_corners = np.ones(edges_im.shape)[(tmp == 682) | (tmp == 597)]


def vectorize_image(im):
    edges_im = detect_edges(im)
    corners_im = detect_corners(edges_im)
    return corners_im
