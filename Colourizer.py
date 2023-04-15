import numpy as np

import FileManager
import Rasterizer


def rgb_to_yiq_mat():
    return np.array([[0.2989, 0.5870, 0.1140],
                     [0.5959, -0.2744, -0.3216],
                     [0.2115, -0.5229, 0.3114]])


def yiq_to_rgb_mat():
    return np.linalg.inv(rgb_to_yiq_mat())


def rgb_to_yiq(im):
    to_yiq_mat = rgb_to_yiq_mat()
    if len(im.shape) == 3:  # Added in case the im.shape is (1920, 1080, 3) and not (1920, 1080).
        return np.dot(im[..., :3], to_yiq_mat)
    return np.dot(im, to_yiq_mat)


def yiq_to_rgb(im):
    to_rgb_mat = yiq_to_rgb_mat()
    return np.dot(im, to_rgb_mat)


def rgb_to_gray(image_frame):
    """
    Calculates the Y channel from an RGB image.
    :param image_frame: An RGB image normalized to [0, 1].
    :return: The Y channel of the converted image.
    """
    return np.dot(image_frame[..., :3], [0.2989, 0.5870, 0.1140])


def alpha_channel(im_y, alpha='n', c=1):
    """
    Generate an alpha channel according to the method indicated for the final image.
    :param im_y: A numpy array with dtype np.float64 in range [0,1]. The y channel of the image.
    :param alpha: A string which represents the method to generate the alpha channel according to. The following are
    possible variables:
    n (non) - no alpha channel, means the alpha image is all 1.
    b (binary) - an image where the alpha is 1 where the y channel is different than 0, and 0 otherwise.
    y (y channel) - an image which is a duplicate of the y channel.
    c (constant) - an image where all its values are equal to the c given as an argument.
    :param c: A float. A constant to multiply the alpha channel by (if given).
    :return: A numpy array with shape equal to the im_y shape and dtype np.float64 in range [0,1].
    """
    if alpha == 'b':  # B for binary.
        return im_y != 0
    elif alpha == 'y':  # Y for y channel (yiq format).
        return im_y
    elif alpha == 'c':  # C for constant coefficient.
        return c * (im_y != 0)
    return np.ones(im_y.shape)


def colour_stroke(stroke, r, g, b, mode='original'):
    c = 1
    if mode == 'random':
        c = np.random.randint(1, 11) * 0.1
    r_stroke = stroke[::, ::, :1:] * r * c
    g_stroke = stroke[::, ::, 1:2:] * g * c
    b_stroke = stroke[::, ::, 2::] * b * c
    coloured_stroke = np.dstack((r_stroke, g_stroke, b_stroke))
    return coloured_stroke


def generate_background(shape):
    output_shape = (720, 1280)  # (1080, 1920)
    c_s = output_shape[0] / 720
    im_i = np.zeros(output_shape)
    im_q = np.zeros(output_shape)
    bcp = np.array([[shape[0], 0], [0.67 * shape[0], 0.3 * shape[1]], [0.3 * shape[0], 0.67 * shape[1]], [0, shape[1]]])
    y_im = Rasterizer.strokes_rasterizer(bcp, 10, 15, canvas_shape=shape, canvas_scalar=1)
    # x = np.linspace(0, shape[0] + 1, shape[0] + 1)
    # y = np.linspace(0, shape[1] + 1, shape[1] + 1)
    # xx, yy = np.meshgrid(x, y)
    # xx += 45
    # yy -= 45
    im_rgb = yiq_to_rgb(y_im)
    im_rgb = colour_stroke(im_rgb, 1.0, 0.49, 0.0, 'original')
    im_alpha = alpha_channel(y_im, alpha='y')
    FileManager.save_rgba_image(FileManager.RAST_DIR_OUT, 'DogColour2', im_rgb, im_alpha)
