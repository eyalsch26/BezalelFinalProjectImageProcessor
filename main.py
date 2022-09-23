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


def t_sparse_vec_check():
    t_orig = np.arange(10 + 1) / 10
    ts_sparse = Rasterizer.t_sparse_vec(t_orig, len(t_orig))
    print(ts_sparse)


def diag_blocks_check():
    arr = np.arange(8).reshape((1, 2, 4))
    r = np.repeat(arr, 3, axis=0)
    print(arr, '\n\n', r)


def rasterizer_check():
    brd = np.zeros(1080 * 1920).reshape((1080, 1920))
    # b_c_p = np.array([[810, 480], [270, 480], [270, 1440], [810, 1440]])
    b_c_p = np.array([[5, 1], [1, 1], [1, 5], [5, 5]])
    b_p = Rasterizer.bezier_curve_rasterizer(b_c_p)
    brd_clrd = np.put_along_axis(brd, b_p, 1, axis=2)
    im_i = np.ones(1080 * 1920).reshape((1080, 1920))
    im_q = np.ones(1080 * 1920).reshape((1080, 1920))
    img = np.dstack((brd_clrd, im_i, im_q))
    im_rgb = np.uint8(255 * Colourizer.yiq_to_rgb(img))
    FileManager.save_image(FileManager.FRAMES_DIR_OUT, im_rgb, 0, 'RasterizerBezierCurve', True)


if __name__ == '__main__':
    # diag_blocks_check()
    # t_sparse_vec_check()
    rasterizer_check()
