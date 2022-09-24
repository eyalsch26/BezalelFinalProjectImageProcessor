import numpy as np
import scipy
from scipy import sparse
from scipy import signal
import matplotlib.pyplot as plt
import FileManager
import EdgeDetector
import Colourizer
import Rasterizer
import Tests


def rasterizer_check():
    brd = np.zeros(1080 * 1920).reshape((1080, 1920))
    b_c_p = np.array([[1, 1], [5, 1], [5, 5], [1, 5]])
    b_p = Rasterizer.bezier_curve_rasterizer(b_c_p)
    # brd_clrd = np.put_along_axis(brd, b_p, 1, axis=2)
    im_i = np.zeros(1080 * 1920).reshape((1080, 1920))
    im_q = np.zeros(1080 * 1920).reshape((1080, 1920))
    img = np.dstack((b_p, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(img))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 0, 'RasterizerBezierCurve', True)


if __name__ == '__main__':
    rasterizer_check()
