import os
import numpy as np
import scipy
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import image
import FileManager, EdgeDetector, Colourizer, Rasterizer, Vectorizer

# Original
FRAMES_DIR_IN = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input'
FRAMES_DIR_OUT = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output'
BLUR_RATIO = 0.25


# def manipulate_image(image_frame, mask_flag=0):
#     new_image_frame = rgb_to_gray(image_frame)
#     if mask_flag == 1:
#         mask = mask_for_fourier(new_image_frame.shape[0], new_image_frame.shape[1], BLUR_RATIO)
#         new_frame = fourier_transform(new_image_frame, mask, mask_flag)  # rgb_to_gray(image_frame)
#     else:
#         new_frame = fourier_transform(new_image_frame)  # rgb_to_gray(image_frame)
#     # new_frame = inverse_fourier_transform(new_frame)
#     return new_frame
#
#
# def manipulate_sequence(directory, start=0, end=1):
#     file_index = 0
#     for file in os.listdir(directory):
#         if file_index < start:
#             file_index += 1
#             continue
#         if file_index > end:
#             break
#         file_name = os.fsdecode(file)
#         image_frame = import_image(FRAMES_DIR_IN + '\\' + str(file_name))
#         new_frame = manipulate_image(image_frame)
#         save_image(FRAMES_DIR_OUT, new_frame, file_index)
#         file_index += 1


# FileManager ----------------------------------------------------------------------------------------------------------
# Works
def sanity_check_rgb_to_yiq_and_back():
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Dog.jpg')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_rgb = (255 * Colourizer.yiq_to_rgb(im_yiq)).astype(np.uint8)
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 1, 'DogRgbToYiqAndBack', False)


# Vectorizer -----------------------------------------------------------------------------------------------------------
# Works.
def sanity_check_rgb_to_yiq_to_fourier_and_back():
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Dog.jpg')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = im_yiq[:, :, 1]
    im_q = im_yiq[:, :, 2]
    fourier_im_y = EdgeDetector.dft2D(im_y)
    im_y_time = EdgeDetector.idft2D(fourier_im_y)
    im_yiq_new = np.dstack((im_y_time, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 2, 'DogRgbToYiqAndBack', True)


# Works.
def sanity_check_fourier_filter():
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Dog.jpg')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = im_yiq[:, :, 1]
    im_q = im_yiq[:, :, 2]
    im_filter = EdgeDetector.gaussian_kernel(23)
    im_filter = EdgeDetector.kernel_padder(im_filter, im_y.shape)
    # Transforming to Fourier domain.
    fourier_im_y = EdgeDetector.dft2D(im_y)
    fourier_filter = EdgeDetector.dft2D(im_filter)
    # Multiplying in Fourier domain.
    fourier_im_y_filtered = fourier_im_y * np.abs(fourier_filter)
    # Returning to time domain.
    im_y_time = EdgeDetector.idft2D(fourier_im_y_filtered)
    im_yiq_new = np.dstack((im_y_time, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 3, 'DogRgbToYiqAndBack', False)


# Works. Does'nt crops the image.
def sanity_check_edge_detection_fourier():
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Aang_Pose_0.0132.jpg')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = im_yiq[:, :, 1]
    im_q = im_yiq[:, :, 2]
    im_filter = EdgeDetector.gaussian_kernel(1)
    im_filter = EdgeDetector.kernel_padder(im_filter, im_y.shape)
    # Transforming to Fourier domain.
    fourier_im_y = EdgeDetector.dft2D(im_y)
    fourier_filter = EdgeDetector.dft2D(im_filter)
    # Multiplying in Fourier domain.
    fourier_im_y_filtered = fourier_im_y * np.abs(fourier_filter)
    # Returning to time domain.
    im_y_blured = EdgeDetector.idft2D(fourier_im_y_filtered)
    im_gradient = np.gradient(im_y_blured)
    im_gradient_x = im_gradient[0]
    im_gradient_y = im_gradient[1]
    im_edges = np.sqrt(np.power(im_gradient_x, 2) + np.power(im_gradient_y, 2))
    im_yiq_new = np.dstack((im_edges, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 0, 'AangGrayEdgesGaussian1', True)


# Works. Crops the image.
def sanity_check_edge_detection_convolution():
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Aang_Pose_0.0132.jpg')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = im_yiq[:, :, 1]
    im_q = im_yiq[:, :, 2]
    im_filter_x = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]]) * 0.5  # EdgeDetector.sobel_kernel('x')
    im_filter_y = im_filter_x.T  #EdgeDetector.sobel_kernel('y')
    im_y = scipy.signal.convolve2d(im_y, EdgeDetector.gaussian_kernel(3), 'same')
    im_y_time_x = scipy.signal.convolve2d(im_y, im_filter_x, 'same')
    im_y_time_y = scipy.signal.convolve2d(im_y, im_filter_y, 'same')
    im_y_time = np.sqrt(np.power(im_y_time_x, 2) + np.power(im_y_time_y, 2))
    # im_yiq_new_x = np.dstack((im_y_time_x, im_i, im_q))
    # im_yiq_new_y = np.dstack((im_y_time_y, im_i, im_q))
    im_yiq_new = np.dstack((im_y_time, im_i, im_q))
    # im_rgb_x = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new_x))
    # im_rgb_y = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new_y))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    # FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb_x, 5, 'AangGrayEdgesX', True)
    # FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb_y, 5, 'AangGrayEdgesY', True)
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 2, 'AangGrayEdgesGradientConvolutionGaussian3', True)


def laplacian_edge_detection_check(t_co):
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Aang_Pose_132_HD720.png')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    im_y_laplacian = Vectorizer.laplacian_image(im_y)
    im_y_laplacian -= np.min(im_y_laplacian)  # Clipping to [0, 1].
    im_y_laplacian = im_y_laplacian / np.max(im_y_laplacian)  # Normalizing.
    t1 = t_co * (np.std(im_y_laplacian) + np.mean(im_y_laplacian))  # np.mean(im_y_laplacian) + t_co * np.std(im_y_laplacian) # t_co * np.mean(im_y_laplacian)
    t2 = t1 * 0.5
    im_y_laplacian[im_y_laplacian >= t1] = 1
    im_y_laplacian[im_y_laplacian < t1] = 0
    im_yiq_new = np.dstack((im_y_laplacian, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 40, f'AangLaplacian{t_co}', True)


def my_edge_detection_check(t1_co, t2_co):
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Aang_Pose_132_HD720.png')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges.
    canny_edges_im = Vectorizer.detect_edges(im_y, t1_co, t2_co)
    im_yiq_new = np.dstack((canny_edges_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 41, f'AangDetectEdgest{t1_co}t{t2_co}HD720', True)


def zero_crossing_check():
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Aang_Pose_132_HD540.png')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the corners image.
    zero_crossing_coordinates = Vectorizer.zero_crossing(im_y)
    im_y_new = np.zeros(im_y.shape)
    im_y_new[zero_crossing_coordinates[0], zero_crossing_coordinates[1]] = 1
    im_yiq_new = np.dstack((im_y_new, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 34, 'AangZeroCrossingHD540', True)


def canny_detector_check(k, g):
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Aang_Pose_132_HD540.png')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges.
    canny_edges_im = Vectorizer.canny_edge_detector(im_y, k=k, gaussian_kernel_size=g)
    im_yiq_new = np.dstack((canny_edges_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 33, 'AangCannyGauss' + str(g) + 'k' + str(k) + 'HD540', True)



def corner_detection_check():
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output'
                                  '\\frame_24_AangGrayEdgesLaplacian.png')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(1080 * 1920).reshape((1080, 1920))
    im_q = np.zeros(1080 * 1920).reshape((1080, 1920))
    # Computing the corners image.
    corner_image = Vectorizer.harris_corner_detector(im_y, 5, 0.04, 0.1)
    im_yiq_new = np.dstack((corner_image, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 27, 'AangCornersFromLaplacian', True)


def corner_detection_sobel_check():
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Aang_Pose_0.0132.jpg')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros((1080, 1920))
    im_q = np.zeros((1080, 1920))
    # Computing the corners image.
    corner_image = Vectorizer.harris_corner_detector_sobel(im_y, 3, 0.04, 0.1)
    im_yiq_new = np.dstack((corner_image, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 30, 'AangCornersSobel', True)


# Rasterizer -----------------------------------------------------------------------------------------------------------
def t_sparse_vec_check():
    t_orig = np.arange(10 + 1) / 10
    ts_sparse = Rasterizer.t_sparse_vec(t_orig, len(t_orig))
    print(ts_sparse)


def diag_blocks_check():
    arr = np.arange(8).reshape((1, 2, 4))
    r = np.repeat(arr, 3, axis=0)
    print(arr, '\n\n', r)


def rasterizer_check():
    brd = np.zeros(1080 * 1920).reshape((1080, 1920))
    b_c_p = np.array([[0, 0], [1079, 0], [1079, 1919], [0, 1919]])
    b_p = Rasterizer.bezier_curve_rasterizer(b_c_p)
    # brd_clrd = np.put_along_axis(brd, b_p, 1, axis=2)
    im_i = np.zeros(1080 * 1920).reshape((1080, 1920))
    im_q = np.zeros(1080 * 1920).reshape((1080, 1920))
    img = np.dstack((b_p, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(img))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 0, 'RasterizerBezierCurve', True)


# def find_contour(image):
#
