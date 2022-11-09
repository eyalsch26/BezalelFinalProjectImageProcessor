import numpy as np


def rgb_to_yiq_mat():
    return np.array([[0.2989, 0.5870, 0.1140],
                     [0.5959, -0.2744, -0.3216],
                     [0.2115, -0.5229, 0.3114]])


def yiq_to_rgb_mat():
    return np.linalg.inv(rgb_to_yiq_mat())


def rgb_to_yiq(im):
    to_yiq_mat = rgb_to_yiq_mat()
    # if len(im.shpae) == 3:  # Added in case the im.shape is (1920, 1080, 3) and not (1920, 1080).
    #     return np.dot(im[..., :3], to_yiq_mat)
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
