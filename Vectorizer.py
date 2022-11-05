import numpy as np


HARRIS_W = 5


def harris_corner_detector(im):
    im_gradient = np.gradient(im)
    ix, iy = im_gradient[0], im_gradient[1]
    ix_square = np.square(ix)
    iy_square = np.square(iy)
    ixiy2 = 2.0 * ix * iy

