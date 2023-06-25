import numpy as np
from matplotlib import pyplot as plt
from imageio.v2 import imread
from matplotlib import image
import Colourizer
import Rasterizer
import Vectorizer
import Renderer


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
RND_TXT_HEAD = '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Text/Headline' \
               '/ParametersFiles_Text_Headline.txt'
RND_CNT_SETUP = '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Content/Setup/ParametersFiles_Content_Setup.txt'

# Constants
FPS = 24


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Importing Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_parameters(path):
    """
    Imports from a given file location the parameters to pass to a renderer function. This is a generic function for
    all renderer function. Each renderer function must verify that it imports the correct number of parameters.
    :param path: String representing the global path to the parameters file.
    :return: A numpy array with dtype float and/or str and with shape (x, 1) where x>0 is the number of parameters
    in the file.
    """
    with open(path, mode='r') as f:
        parameters = np.array([l.splitlines() for l in f.readlines()[1::2]]).flatten()
        parameters = [float(p) if p.isdigit() else str(p) for p in parameters]
    f.close()
    return parameters


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Importing & Saving Images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def file_path(dir_path, file_name, num, ext='png', os='w', txt=False):
    f_path = f'{dir_path}\\{file_name}.{num}.{ext}'  # Windows.
    if txt:
        f_path = f'{dir_path}\\{file_name}.{ext}'  # Windows.
    if os == 'm':  # Mac.
        f_path = f'{dir_path}/{file_name}.{num}.{ext}'
        if txt:
            f_path = f'{dir_path}/{file_name}.{ext}'
    return f_path


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


def save_rgba_image(dir_path, f_name, im_rgb, im_alpha, os='w'):
    # Naming the path of the new frame to be saved.
    path = dir_path + '\\' + f_name + '.png'
    im_rgba = (255 * np.clip(np.dstack((im_rgb, im_alpha)), 0, 1)).astype(np.uint8)
    if os == 'm':
        path = dir_path + '/' + f_name + '.png'
        im_rgba = (255 * np.clip(np.dstack((im_rgb, im_alpha)), 0, 1)).astype(np.uint8).copy(order='C')
    # Saving the path.
    plt.imsave(path, im_rgba)
    return


def save_rgba_image0(dir_path, f_name, im_rgb, im_alpha):
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
    """
    Extracts a numpy array of Bezier control points from a file. As a convention each file holds Bezier curves from a
    specific image so cuvers for different images will be stored in different files.
    :param path: String. The location of the file to extract the Bezier control points from (the name of the specific
    file is included in the path).
    :return: A numpy array with dtype np.float64 and with shape (x, 4, 2) where x>0 represents the number of curves
    in the file.
    """
    raw_data = np.loadtxt(path)
    bzr_ctrl_pts_num = int(len(raw_data) * 0.25)
    data = raw_data.reshape((bzr_ctrl_pts_num, 4, 2))
    return data, bzr_ctrl_pts_num


def save_bezier_control_points(path, bzr_ctrl_pts_arr):
    with open(path, mode='w') as output_file:
        for single_curve in bzr_ctrl_pts_arr:
            np.savetxt(output_file, single_curve, fmt='%-50.32f')
            output_file.write('#\n')
    output_file.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rendering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def contour(in_path, out_path, bcp_path, start, end, digits_num, dst_f, dst_s, stk_min, stk_max, stk_styl):
    n = end - start + 1  # Adding 1 to include the last index.
    for im_file_idx in range(1, n + 1):
        # Preparing the output file name.
        n_padded = f'{im_file_idx}'
        while (len(n_padded) < digits_num):
            n_padded = f'0{n_padded}'
        # Preparing the image.
        im_name = f'{in_path}{n_padded}.png'
        im = import_image(im_name)
        # Finding the Bezier control points.
        im_bcp = Vectorizer.vectorize_image(im)
        # Saving the Bezier control points to a file.
        bcp_f_name = f'{bcp_path}{n_padded}.txt'
        save_bezier_control_points(bcp_f_name, im_bcp)
        # Reading the Bezier control points from files.

        # Rasterizing the curves.

        # Saving the images.


# Content
def vectorize_content_to_file(in_path, bcp_path, start, end, digits_num):
    n = end - start + 1  # Adding 1 to include the last index.
    # Iterating over the desired images.
    for im_file_idx in range(1, n + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while (len(n_padded) < digits_num):
            n_padded = f'0{n_padded}'
        # Preparing the image.
        im_name = f'{in_path}{n_padded}.png'
        im = import_image(im_name)
        # Finding the Bezier control points.
        im_bcp = Vectorizer.vectorize_image(im)
        # Saving the Bezier control points to a file.
        bcp_f_name = f'{bcp_path}{n_padded}.txt'
        save_bezier_control_points(bcp_f_name, im_bcp)


def raster_content_from_file(prm_path, bcp_path, out_path, start, end, digits_num, txr, clr, rgb_range):
    # Importing the parameters for the functions.
    prm = import_parameters(prm_path)
    n = end - start + 1  # Adding 1 to include the last index.
    # Iterating over the desired images.
    for im_file_idx in range(1, n + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while (len(n_padded) < digits_num):
            n_padded = f'0{n_padded}'
        # Importing the Bezier control points from the text file.
        bcp_f_name = f'{bcp_path}{n_padded}.txt'
        bcp_arr = import_bezier_control_points(bcp_f_name)
        # Rastering the curves.
        curves_num = len(bcp_arr)
        txr_arr = Rasterizer.generate_textures_arr(curves_num, txr)  # Defining texture.
        clr_arr = Colourizer.generate_colours_arr(curves_num, clr, rgb_range)  # Defining colour.
        for crv_idx in range(curves_num):
            cur_bcp = bcp_arr[crv_idx]
            cur_txr = txr_arr[crv_idx]
            cur_clr = clr_arr[crv_idx]

