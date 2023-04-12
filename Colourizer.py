import numpy as np


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
