import time
import os
import numpy as np
import scipy
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import image
import FileManager, EdgeDetector, Colourizer, Rasterizer, Vectorizer

# Original
FRAMES_DIR_IN = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input'
VEC_DIR_OUT = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Vector'
RAST_DIR_OUT = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Raster'
BLUR_RATIO = 0.25

# --------------------------------------------------- Hot Keys ---------------------------------------------------------
# Ctrl + B : Go to declaration/usage of function.
# Ctrl + Alt + M : Generate function from statement. (Old: Alt + Enter).
# Ctrl + Shift + Up/Down : Moves the line(s) up or down respectively.
# Lambda expression: a = x if b < c else y.
# ----------------------------------------------------------------------------------------------------------------------


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
    FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 1, 'DogRgbToYiqAndBack', False)


# Works.
def write_bzr_ctrl_pts_to_file_check():
    # Preparing the image and the filter.
    im = FileManager.import_image(FileManager.FRAME_IN)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    # Computing the image's edges' bezier control points.
    bzr_ctrl_pts_arr = Vectorizer.vectorize_image(im_y)
    FileManager.save_bezier_control_points(FileManager.TEXT_DIR + '\\test2.txt', bzr_ctrl_pts_arr)


# Works.
def read_bzr_ctrl_pts_from_file_check():
    output_shape = (720, 1280)  # (1080, 1920)
    c_s = output_shape[0]/720
    im_i = np.zeros(output_shape)
    im_q = np.zeros(output_shape)
    bzr_ctrl_pts_arr = FileManager.import_bezier_control_points(FileManager.TEXT_DIR + '\\test2.txt')
    raster_im = Rasterizer.strokes_rasterizer(bzr_ctrl_pts_arr, canvas_shape=output_shape, canvas_scalar=c_s)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    im_alpha = Colourizer.alpha_channel(raster_im, alpha='y')
    FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, 'DogFromFileDisplacement10', im_rgb, im_alpha)


# Vectorizer -----------------------------------------------------------------------------------------------------------
def pixels_count():
    # Preparing the image and the filter.
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Vector'
                                  '\\frame_53_Dog02VectorizeRecoverThreshold015HD720.png')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    print(len(np.argwhere(im_y > 0)))


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
    FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 2, 'DogRgbToYiqAndBack', True)


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
    FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 3, 'DogRgbToYiqAndBack', False)


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
    FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 0, 'AangGrayEdgesGaussian1', True)


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
    FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 2, 'AangGrayEdgesGradientConvolutionGaussian3', True)


# Works. (In current state: t1 = t_co * (np.std(im_y_laplacian) + np.mean(im_y_laplacian))).
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
    FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 40, f'AangLaplacian{t_co}', True)


# Works. Edges are not necessarily one pixel wide (but improved).
def my_edge_detection_check(t1_co=0.975, t2_co=0.995):
    # Preparing the image and the filter.
    im = FileManager.import_image(FileManager.FRAME_IN_MAC)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges.
    # canny_edges_im = Vectorizer.detect_edges(im_y, t1_co, t2_co)  # Original.
    canny_edges_im = Vectorizer.detect_edges_new(im_y)
    im_yiq_new = np.dstack((canny_edges_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    # FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 51, f'Dog_01DetectEdgesT{t1_co}T{t2_co}', True)  # Original.
    FileManager.save_image(FileManager.VEC_DIR_OUT_MAC, im_rgb, 2, f'DetectEdges', True, 'm')


def my_corner_detection_check(t1_co=0.975, t2_co=0.995):
    # Preparing the image and the filter.
    im = FileManager.import_image(FileManager.FRAME_IN_MAC)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges.
    canny_edges_im = Vectorizer.detect_edges_new(im_y)
    corners_im = Vectorizer.detect_corners(canny_edges_im)
    corners_on_edges_im = (canny_edges_im + corners_im) * 0.5
    im_yiq_new = np.dstack((corners_on_edges_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.VEC_DIR_OUT_MAC, im_rgb, 15, f'DetectCorners', True, 'm')


# Works.
def trace_edge_from_corner_basic_check():
    edges_im = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0]])
    corner_im = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0]])
    p_0 = np.array([1, 2])
    paths = Vectorizer.trace_edge_from_corner(edges_im, corner_im, p_0)
    print(paths)


# Works.
def vectorize_check():
    # Preparing the image and the filter.
    im = FileManager.import_image(FileManager.FRAME_IN)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges.
    bzr_ctrl_pts_arr = Vectorizer.vectorize_image(im_y)
    raster_im = Rasterizer.bezier_curves_rasterizer(bzr_ctrl_pts_arr, canvas_shape=im_y.shape)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 56, f'Dog02VectorizeHD720', True)


# Works.
def displacement_check(in_path, out_path):
    # Preparing the image and the filter.
    im = FileManager.import_image(in_path)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges' bezier control points.
    bzr_ctrl_pts_arr = Vectorizer.vectorize_image(im_y)
    new_bzr_ctrl_pts = Vectorizer.displace_bezier_curves(bzr_ctrl_pts_arr)
    raster_im = Rasterizer.bezier_curves_rasterizer(new_bzr_ctrl_pts, canvas_shape=im_y.shape)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(out_path, im_rgb, 4, f'DisplacementHD720', True)


def show_reel(in_path, out_path, start, end, zero_pad):
    n = end - start
    for im_file_idx in range(1, n+1):
        n_padded = f'0{im_file_idx}'
        while (len(n_padded) < zero_pad):
            n_padded = f'0{n_padded}'
        # Preparing the image and the filter.
        im = FileManager.import_image(f'{in_path}{n_padded}.png')
        im_yiq = Colourizer.rgb_to_yiq(im)
        im_y = im_yiq[:, :, 0]
        im_i = np.zeros(im_y.shape)
        im_q = np.zeros(im_y.shape)
        # Computing the image's edges.
        corners_im = Vectorizer.vectorize_image(im_y)
        im_yiq_new = np.dstack((corners_im, im_i, im_q))
        im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
        image_frame = Colourizer.rgb_to_gray(im_rgb)
        plt.imsave(out_path + n_padded + '.png', image_frame, cmap='gray')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mac ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Works.
def vectorize_check_mac():
    # Preparing the image and the filter.
    im = FileManager.import_image(FileManager.FRAME_IN_MAC)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges.
    bzr_ctrl_pts_arr = Vectorizer.vectorize_image_new(im_y)
    raster_im = Rasterizer.bezier_curves_rasterizer(bzr_ctrl_pts_arr, canvas_shape=im_y.shape)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.VEC_DIR_OUT_MAC, im_rgb, 10, f'Vectorize', True, 'm')


def trace_edges_check():
    # Preparing the image and the filter.
    im = FileManager.import_image(FileManager.FRAME_IN_MAC)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges.
    edges_im = Vectorizer.detect_edges_new(im_y)
    corners_im = Vectorizer.detect_corners(edges_im)
    paths_dict = Vectorizer.trace_edges_to_paths_new(edges_im, corners_im, 45)
    paths_num = len(paths_dict)
    for p in range(paths_num):
        new_y_im = np.zeros(im_y.shape)
        rows = paths_dict[p].T[0].astype(int)
        columns = paths_dict[p].T[1].astype(int)
        new_y_im[rows, columns] = 1
        im_yiq_new = np.dstack((new_y_im, im_i, im_q))
        im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
        FileManager.save_image(FileManager.VEC_DIR_OUT_MAC, im_rgb, 4, f'TracePaths{p}', True, 'm')


# Works.
def displacement_check_mac(in_path, out_path, frames_num):
    # Preparing the image and the filter.
    im = FileManager.import_image(in_path)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges' bezier control points.
    bzr_ctrl_pts_arr = Vectorizer.vectorize_image(im_y)
    new_bzr_ctrl_pts = Vectorizer.displace_bezier_curves(bzr_ctrl_pts_arr)
    raster_im = Rasterizer.bezier_curves_rasterizer(new_bzr_ctrl_pts, canvas_shape=im_y.shape)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(out_path, im_rgb, 00, f'DisplacementHD720', True)


# Works.
def displacement_sequence_check_mac(in_path, out_path, frames_num):
    # Preparing the image and the filter.
    im = FileManager.import_image(in_path)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges' bezier control points.
    bzr_ctrl_pts_arr = Vectorizer.vectorize_image(im_y)
    for i in range(frames_num):
        new_bzr_ctrl_pts = Vectorizer.displace_bezier_curves(bzr_ctrl_pts_arr)
        raster_im = Rasterizer.bezier_curves_rasterizer(new_bzr_ctrl_pts, canvas_shape=im_y.shape)
        im_yiq_new = np.dstack((raster_im, im_i, im_q))
        im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
        FileManager.save_image(out_path, im_rgb, 00, f'Displacement{i}HD720', True)



def show_reel_mac(in_path, out_path, start, end, zero_pad):
    n = end - start
    for im_file_idx in range(1, n+1):
        # Preparing the output file name.
        n_padded = f'0{im_file_idx}'
        while (len(n_padded) < zero_pad):
            n_padded = f'0{n_padded}'
        # Preparing the image and the filter.
        im = FileManager.import_image(f'{in_path}{n_padded}.png')
        im_yiq = Colourizer.rgb_to_yiq(im)
        im_y = im_yiq[:, :, 0]
        im_i = np.zeros(im_y.shape)
        im_q = np.zeros(im_y.shape)
        # Computing the image's edges.
        corners_im = Vectorizer.vectorize_image(im_y)
        im_yiq_new = np.dstack((corners_im, im_i, im_q))
        im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
        image_frame = Colourizer.rgb_to_gray(im_rgb)
        plt.imsave(out_path + n_padded + '.png', image_frame, cmap='gray')


def volume_colourizer_check_mac():
    # Preparing the image and the filter.
    im = FileManager.import_image(FileManager.FRAME_IN_MAC)
    r = np.max(im[::, ::, 0])
    g = np.max(im[::, ::, 1])
    b = np.max(im[::, ::, 2])
    a = Vectorizer.blur_image(Colourizer.alpha_channel(im[::, ::, 3], 'r', 2), 9)
    shape = im[:, :, 0].shape
    # a = Vectorizer.blur_image(im[::, ::, 3] * np.random.randint(0, 255, shape) / 255, 7)
    # im_rgb = Colourizer.colour_volume(shape, 1.0, 0.49, 0, 50, 9, 'noise')
    im_rgb = Colourizer.colour_volume(shape, r, g, b, 50, 9, 'noise')
    FileManager.save_rgba_image_mac(FileManager.CLR_DIR_OUT_MAC, '19VolumeColourHD720', im_rgb, a)


def colour_image_check_mac():
    im = FileManager.import_image(FileManager.FRAME_IN_MAC)
    im_yiq = Colourizer.rgb_to_yiq(im)
    y_im = im_yiq[:, :, 0]
    im_rgb_cur = Colourizer.yiq_to_rgb(im_yiq)
    im_rgb = Colourizer.colour_stroke(im_rgb_cur, 1.0, 0.49, 0.0, 'original')
    im_alpha = Colourizer.alpha_channel(y_im, alpha='y')
    FileManager.save_rgba_image(FileManager.CLR_DIR_OUT_MAC, 'DogColour2', im_rgb, im_alpha)


# Rasterizer -----------------------------------------------------------------------------------------------------------
def t_sparse_vec_check():
    t_orig = np.arange(10 + 1) / 10
    ts_sparse = Rasterizer.t_sparse_vec(t_orig)
    print(ts_sparse)


def diag_blocks_check():
    arr = np.arange(8).reshape((1, 2, 4))
    r = np.repeat(arr, 3, axis=0)
    print(arr, '\n\n', r)


def rasterizer_check():
    brd = np.zeros(1080 * 1920).reshape((1080, 1920))
    # b_c_p = np.array([[539, 957], [541, 958.5], [538, 960.5], [540, 962]])
    # b_c_p = np.array([[540, 960], [270, 480], [270, 1440], [540, 960]])
    b_c_p = np.array([[0, 0], [1079, 0], [1079, 1919], [0, 1919]])
    b_p = Rasterizer.bezier_curve_rasterizer(b_c_p)
    # brd_clrd = np.put_along_axis(brd, b_p, 1, axis=2)
    im_i = np.zeros(1080 * 1920).reshape((1080, 1920))
    im_q = np.zeros(1080 * 1920).reshape((1080, 1920))
    img = np.dstack((b_p, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(img))
    FileManager.save_image(FileManager.RAST_DIR_OUT, im_rgb, 5, 'RasterizerBezierCurve', True)


def stroke_rasterizer_check():
    b_c_p = np.array([[270, 480], [810, 480], [810, 1440], [270, 1440]])
    b_p = Rasterizer.stroke_rasterizer(b_c_p, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                       strength_style='uniform')
    im_i = np.zeros(1080 * 1920).reshape((1080, 1920))
    im_q = np.zeros(1080 * 1920).reshape((1080, 1920))
    img = np.dstack((b_p, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(img))
    FileManager.save_image(FileManager.RAST_DIR_OUT, im_rgb, 9, f'RasterizerStroke', True)


def face_stroke_rasterizer_check():
    b_c_p_0 = np.array([[404, 719], [39, 709], [9, 1219], [404, 1199]])
    b_c_p_1 = np.array([[424, 724], [639, 669], [699, 729], [863, 819]])
    b_c_p_2 = np.array([[873, 829], [909, 824], [924, 1129], [579, 1139]])
    b_c_p_3 = np.array([[459, 1169], [404, 1199], [479, 1304], [639, 1164]])
    b_c_p_4 = np.array([[434, 889], [399, 939], [444, 989], [464, 1059]])
    b_c_p_5 = np.array([[414, 722], [394, 744], [389, 839], [429, 839]])
    b_c_p_6 = np.array([[434, 839], [704, 754], [719, 784], [654, 904]])
    b_c_p_7 = np.array([[724, 794], [734, 809], [734, 909], [714, 974]])
    b_c_p_8 = np.array([[519, 954], [464, 904], [464, 1014], [514, 959]])
    b_p_0 = Rasterizer.stroke_rasterizer(b_c_p_0, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p_1 = Rasterizer.stroke_rasterizer(b_c_p_1, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p_2 = Rasterizer.stroke_rasterizer(b_c_p_2, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p_3 = Rasterizer.stroke_rasterizer(b_c_p_3, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p_4 = Rasterizer.stroke_rasterizer(b_c_p_4, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p_5 = Rasterizer.stroke_rasterizer(b_c_p_5, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p_6 = Rasterizer.stroke_rasterizer(b_c_p_6, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p_7 = Rasterizer.stroke_rasterizer(b_c_p_7, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p_8 = Rasterizer.stroke_rasterizer(b_c_p_8, radius_min=2, radius_max=3, texture='random', blur_kernel=3,
                                         strength_style='uniform')
    b_p = np.clip(b_p_0 + b_p_1 + b_p_2 + b_p_3 + b_p_4 + b_p_5 + b_p_6 + b_p_7 + b_p_8, 0, 1)
    im_i = np.zeros((1080, 1920))
    im_q = np.zeros((1080, 1920))
    img = np.dstack((b_p, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(img))
    FileManager.save_image(FileManager.RAST_DIR_OUT, im_rgb, 9, f'RasterizerStrokeFace', True)


def sequence_face_stroke_rasterizer_check():
    b_c_p_0 = np.array([[404, 719], [39, 709], [9, 1219], [404, 1199]])
    b_c_p_1 = np.array([[424, 724], [639, 669], [699, 729], [863, 819]])
    b_c_p_2 = np.array([[873, 829], [909, 824], [924, 1129], [579, 1139]])
    b_c_p_3 = np.array([[459, 1169], [404, 1199], [479, 1304], [639, 1164]])
    b_c_p_4 = np.array([[434, 889], [399, 939], [444, 989], [464, 1059]])
    b_c_p_5 = np.array([[414, 722], [394, 744], [389, 839], [429, 839]])
    b_c_p_6 = np.array([[434, 839], [704, 754], [719, 784], [654, 904]])
    b_c_p_7 = np.array([[724, 794], [734, 809], [734, 909], [714, 974]])
    b_c_p_8 = np.array([[519, 954], [464, 904], [464, 1014], [514, 959]])
    b_c_p_arr = np.array([b_c_p_0, b_c_p_1, b_c_p_2, b_c_p_3, b_c_p_4, b_c_p_5, b_c_p_6, b_c_p_7, b_c_p_8])
    strokes_num = 9
    sec = 5
    fps = 24
    for s in range(sec):
        for f in range(fps):
            r_v = np.random.randint(-5, 5, size=(strokes_num, 4, 2))
            r_m = np.random.randint(1, 3, size=strokes_num)
            r_d = np.random.randint(1, 3, size=strokes_num)
            b_c = np.zeros((1080, 1920))
            for i in range(strokes_num):
                cur_b_c_p = b_c_p_arr[i] + r_v[i]
                cur_b_c = Rasterizer.stroke_rasterizer(cur_b_c_p, radius_min=r_m[i], radius_max=(r_m[i] + r_d[i]),
                                                       texture='random', blur_kernel=3, strength_style='uniform')
                b_c += cur_b_c
            b_c = np.clip(b_c, 0, 1)
            im_i = np.zeros((1080, 1920))
            im_q = np.zeros((1080, 1920))
            img = np.dstack((b_c, im_i, im_q))
            im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(img))
            f_s = f'0{f}' if len(str(f)) < 2 else str(f)
            FileManager.save_image(FileManager.RAST_DIR_OUT, im_rgb, 0, f'FaceSequence{s}{f_s}', True)
    return


def vector_strokes_check():
    # Preparing the image and the filter.
    im = FileManager.import_image(FileManager.FRAME_IN)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(im_y.shape)
    im_q = np.zeros(im_y.shape)
    # Computing the image's edges.
    bzr_ctrl_pts_arr = Vectorizer.vectorize_image(im_y)
    raster_im = Rasterizer.strokes_rasterizer(bzr_ctrl_pts_arr, canvas_shape=im_y.shape)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    FileManager.save_image(FileManager.VEC_DIR_OUT, im_rgb, 55, f'Dog02VectorStrokesHD720', True)


def vector_strokes_displacement_check(in_path, out_path):
    # Preparing the image and the filter.
    output_shape = (1080, 1920)
    im = FileManager.import_image(in_path)
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_y = im_yiq[:, :, 0]
    im_i = np.zeros(output_shape)
    im_q = np.zeros(output_shape)
    # Computing the image's edges' bezier control points.
    bzr_ctrl_pts_arr = Vectorizer.vectorize_image(im_y)
    new_bzr_ctrl_pts = Vectorizer.displace_bezier_curves(bzr_ctrl_pts_arr)
    raster_im = Rasterizer.strokes_rasterizer(new_bzr_ctrl_pts, canvas_shape=output_shape, canvas_scalar=1.5)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    im_alpha = Colourizer.alpha_channel(raster_im, alpha='y')
    FileManager.save_rgba_image(out_path, 'DogDisplacementEnlarged0', im_rgb, im_alpha)


# Works.
def save_rgba_check():
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Raster\Dog'
                                  '\\frame_2_StrokesDisplacementHD720.png')
    im_yiq = Colourizer.rgb_to_yiq(im)
    im_alpha = Colourizer.alpha_channel(im_yiq[:, :, 0], alpha='y')  # Original: im_yiq[:, :, 0]
    im_rgb = Colourizer.yiq_to_rgb(im_yiq)
    # im_rgba = (255 * np.dstack((im_rgb, im_alpha))).astype(np.uint8)  # Original.
    FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, 'DogAlphaY1', im_rgb, im_alpha)


# Works.
def displace_bcp_from_file_check():
    output_shape = (720, 1280)  # (1080, 1920)
    c_s = output_shape[0]/720
    im_i = np.zeros(output_shape)
    im_q = np.zeros(output_shape)
    bzr_ctrl_pts_arr = FileManager.import_bezier_control_points(FileManager.TEXT_DIR + '\\test2.txt')
    new_bzr_ctrl_pts = Vectorizer.displace_bezier_curves(bzr_ctrl_pts_arr)
    raster_im = Rasterizer.strokes_rasterizer(new_bzr_ctrl_pts, canvas_shape=output_shape, canvas_scalar=c_s)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    im_alpha = Colourizer.alpha_channel(raster_im, alpha='y')
    FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, 'DogFromFileDisplacement16', im_rgb, im_alpha)


# Works.
def displace_sequence_bcp_from_file_check(frames_num):
    t_s = time.time()
    output_shape = (720, 1280)  # (1080, 1920)
    c_s = output_shape[0]/720
    im_i = np.zeros(output_shape)
    im_q = np.zeros(output_shape)
    bzr_ctrl_pts_arr = FileManager.import_bezier_control_points(FileManager.TEXT_DIR + '\\test2.txt')
    for i in range(frames_num):
        f = (i // 4) + 1
        new_bzr_ctrl_pts = Vectorizer.displace_bezier_curves(bzr_ctrl_pts_arr, f, 2 * f)
        raster_im = Rasterizer.strokes_rasterizer(new_bzr_ctrl_pts, canvas_shape=output_shape, canvas_scalar=c_s)
        im_yiq_new = np.dstack((raster_im, im_i, im_q))
        im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
        im_alpha = Colourizer.alpha_channel(raster_im, alpha='y')
        FileManager.save_bezier_control_points(FileManager.TEXT_DIR + f'\\DogDisplaceBCP{i}.txt', new_bzr_ctrl_pts)
        FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, f'DogFromFileDisplacement{i}', im_rgb, im_alpha)
    t_e = time.time()
    print(t_e - t_s)  # Date: 10.4.2023: 1494.8511474132538 (which is ~20.75 seconds per frame).


# Works.
def distort_bcp_from_file_check():
    output_shape = (720, 1280)  # (1080, 1920)
    c_s = output_shape[0]/720
    im_i = np.zeros(output_shape)
    im_q = np.zeros(output_shape)
    bzr_ctrl_pts_arr = FileManager.import_bezier_control_points(FileManager.TEXT_DIR + '\\test2.txt')
    new_bzr_ctrl_pts = Vectorizer.distort_bezier_curves(bzr_ctrl_pts_arr, 10)
    raster_im = Rasterizer.strokes_rasterizer(new_bzr_ctrl_pts, canvas_shape=output_shape, canvas_scalar=c_s)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    im_alpha = Colourizer.alpha_channel(raster_im, alpha='y')
    FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, 'DogFromFileDisplacement19', im_rgb, im_alpha)


# Works.
def displace_distort_bcp_from_file_check():
    output_shape = (720, 1280)  # (1080, 1920)
    c_s = output_shape[0]/720
    im_i = np.zeros(output_shape)
    im_q = np.zeros(output_shape)
    bzr_ctrl_pts_arr = FileManager.import_bezier_control_points(FileManager.TEXT_DIR + '\\test2.txt')
    new_bzr_ctrl_pts = Vectorizer.displace_bezier_curves(bzr_ctrl_pts_arr, 42, 12)
    new_bzr_ctrl_pts = Vectorizer.distort_bezier_curves(new_bzr_ctrl_pts, 8)
    raster_im = Rasterizer.strokes_rasterizer(new_bzr_ctrl_pts, canvas_shape=output_shape, canvas_scalar=c_s)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(im_yiq_new))
    im_alpha = Colourizer.alpha_channel(raster_im, alpha='y')
    FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, 'DogFromFileDisplacement34', im_rgb, im_alpha)


# Works.
def colour_image_check():
    im = FileManager.import_image('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Raster'
                                  '\DogFromFileDisplacement27.png')
    im_yiq = Colourizer.rgb_to_yiq(im)
    y_im = im_yiq[:, :, 0]
    im_rgb_cur = Colourizer.yiq_to_rgb(im_yiq)
    im_rgb = Colourizer.colour_stroke(im_rgb_cur, 1.0, 0.49, 0.0, 'original')
    im_alpha = Colourizer.alpha_channel(y_im, alpha='y')
    FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, 'DogColour2', im_rgb, im_alpha)


def displace_distort_colour_bcp_from_file_check():
    output_shape = (720, 1280)  # (1080, 1920)
    c_s = output_shape[0]/720
    im_i = np.zeros(output_shape)
    im_q = np.zeros(output_shape)
    bzr_ctrl_pts_arr = FileManager.import_bezier_control_points(FileManager.TEXT_DIR + '\\test2.txt')
    new_bzr_ctrl_pts = Vectorizer.displace_bezier_curves(bzr_ctrl_pts_arr, 42, 12)
    new_bzr_ctrl_pts = Vectorizer.distort_bezier_curves(new_bzr_ctrl_pts, 8)
    raster_im = Rasterizer.strokes_rasterizer(new_bzr_ctrl_pts, 3, 10, canvas_shape=output_shape, canvas_scalar=c_s)
    im_yiq_new = np.dstack((raster_im, im_i, im_q))
    im_rgb_cur = Colourizer.yiq_to_rgb(im_yiq_new)
    im_rgb = Colourizer.colour_stroke(im_rgb_cur, 1.0, 0.49, 0)
    im_alpha = Colourizer.alpha_channel(raster_im, alpha='b')
    FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, 'DogFromFileDisplacement38', im_rgb, im_alpha)
