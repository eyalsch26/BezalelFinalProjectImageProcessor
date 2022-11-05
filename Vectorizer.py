import numpy as np


def harris_corner_detector(im):
    im_gradient = np.gradient(im)
