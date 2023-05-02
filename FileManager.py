import os
import numpy as np
from matplotlib import pyplot as plt
from imageio.v2 import imread
from matplotlib import image
import Colourizer
import Vectorizer


# Windows
FRAMES_DIR_IN = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input'
VEC_DIR_OUT = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Vector'
RAST_DIR_OUT = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Raster'
FRAME_IN = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Dog_02.png'
TEXT_DIR = 'G:\Eyal\Documents\Bezalel\FinalProject\BCPData'
# Mac
FRAME_IN_MAC = '/Users/eyalschaffer/Pictures/BezalelFinalProject/Input/Shape.01.png'
FRAMES_DIR_IN_MAC = '/Users/eyalschaffer/Documents/maya/projects/A_Moment_In_Life/images/Shape.01.png'
VEC_DIR_OUT_MAC = '/Users/eyalschaffer/Pictures/BezalelFinalProject/Output/Vector'
RAST_DIR_OUT_MAC = '/Users/eyalschaffer/Pictures/BezalelFinalProject/Output/Raster'
CLR_DIR_OUT_MAC = '/Users/eyalschaffer/Pictures/BezalelFinalProject/Output/Colour'

# Constants
FPS = 24


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Importing & Saving Images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_image(path):
    frame = imread(path).astype(np.float64)
    if np.amax(frame) > 1:
        frame /= 255
    return frame


def save_image(dir_path, image_frame, frame_index, name_suffix, as_grayscale=True):
    # Naming the path of the new frame to be saved.
    path = dir_path + '\\frame_' + str(frame_index) + '_' + name_suffix + '.png'
    # Saving the path.
    if as_grayscale:
        image_frame = Colourizer.rgb_to_gray(image_frame)
        plt.imsave(path, image_frame, cmap='gray')
    else:
        plt.imsave(path, image_frame)
    return


def save_rgba_image(dir_path, f_name, im_rgb, im_alpha):
    # Naming the path of the new frame to be saved.
    path = dir_path + '\\' + f_name + '.png'
    im_rgba = (255 * np.clip(np.dstack((im_rgb, im_alpha)), 0, 1)).astype(np.uint8)
    # Saving the path.
    plt.imsave(path, im_rgba)
    return


def save_rgba_image_mac(dir_path, f_name, im_rgb, im_alpha):
    # Naming the path of the new frame to be saved.
    path = dir_path + '/' + f_name + '.png'
    im_rgba = (255 * np.clip(np.dstack((im_rgb, im_alpha)), 0, 1)).astype(np.uint8).copy(order='C')
    # Saving the path.
    plt.imsave(path, im_rgba)
    return


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Importing & Saving Vectors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_bezier_control_points(path):
    raw_data = np.loadtxt(path)
    bzr_ctrl_pts_num = int(len(raw_data) * 0.25)
    data = raw_data.reshape((bzr_ctrl_pts_num, 4, 2))
    return data


def save_bezier_control_points(path, bzr_ctrl_pts_arr):
    with open(path, mode='w') as output_file:
        for single_curve in bzr_ctrl_pts_arr:
            np.savetxt(output_file, single_curve, fmt='%-50.32f')
            output_file.write('#\n')
    output_file.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rendering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def volume_colourizer(in_path, out_path, start, end, digits_num, alpha_t, c, blr_k, tlr, clr_s):
    n = end - start + 1  # Adding 1 to include the last index.
    for im_file_idx in range(1, n+1):
        # Preparing the output file name.
        n_padded = f'{im_file_idx}'
        while (len(n_padded) < digits_num):
            n_padded = f'0{n_padded}'
        # Preparing the image and the filter.
        im_name = f'{in_path}{n_padded}.png'
        im = import_image(im_name)
        r = np.max(im[::, ::, 0])
        g = np.max(im[::, ::, 1])
        b = np.max(im[::, ::, 2])
        a = Vectorizer.blur_image(Colourizer.alpha_channel(im[::, ::, 3], alpha_t, c), blr_k)
        shape = im[:, :, 0].shape
        im_rgb = Colourizer.colour_volume(shape, r, g, b, tlr, blr_k, clr_s)
        save_rgba_image(out_path, im_name + 'Rnd', im_rgb, a)


def contour(in_path, out_path, start, end, digits_num, dst_f, dst_s, stk_min, stk_max, stk_styl):
    n = end - start + 1  # Adding 1 to include the last index.
    for im_file_idx in range(1, n + 1):
        # Preparing the output file name.
        n_padded = f'{im_file_idx}'
        while (len(n_padded) < digits_num):
            n_padded = f'0{n_padded}'
        # Preparing the image.
        im_name = f'{in_path}{n_padded}.png'
        im = import_image(im_name)
        # Finding the Bezier control points and saving to files.

        # Reading the Bezier control points from files.

        # Rasterizing the curves.

        # Saving the images.
