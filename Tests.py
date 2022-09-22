import os
import numpy as np
import scipy
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import image
import FileManager, EdgeDetector, Colourizer

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


# Works
def sanity_check_rgb_to_yiq_and_back():
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Dog.jpg')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_rgb = (255 * Colourizer.yiq_to_rgb(im_yiq)).astype(np.uint8)
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 1, 'DogRgbToYiqAndBack', False)


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


# def find_contour(image):
#
