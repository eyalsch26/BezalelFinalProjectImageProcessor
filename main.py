import numpy as np
import scipy
from scipy import ndimage
from scipy import sparse
from scipy import signal
import matplotlib.pyplot as plt
import FileManager
import EdgeDetector
import Colourizer
import Rasterizer
import Tests


if __name__ == '__main__':
    # a = np.zeros((2, 5, 5))
    # b = np.arange(25).reshape((5, 5))
    # print(b.shape)
    # # b = ndimage.maximum_filter(b, footprint=np.ones((3, 3)))
    # print(b.size)
    # Tests.laplacian_edge_detection()
    Tests.canny_detector_check()
