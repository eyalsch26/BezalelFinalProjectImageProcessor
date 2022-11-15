import numpy as np
import scipy
from scipy import ndimage
from scipy import sparse
from scipy import signal
import matplotlib.pyplot as plt
import FileManager
import EdgeDetector
import Vectorizer
import Colourizer
import Rasterizer
import Tests


if __name__ == '__main__':
    # a = np.zeros((2, 5, 5))
    # b = np.arange(25).reshape((5, 5))
    # b = ndimage.maximum_filter(b, footprint=np.ones((3, 3)))
    # z = (b > 5) & (b < 20)
    # print(b)
    # print(z * b)
    # Tests.zero_crossing_check()
    Tests.my_edge_detection_check(0.975, 0.995)
    # Tests.laplacian_edge_detection_check(0.9725)
    # Tests.canny_detector_check(20, 7)
