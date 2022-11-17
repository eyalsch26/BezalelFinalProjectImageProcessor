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
    # a = Vectorizer.blur_image(np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]]), 3)
    # b = np.arange(25).reshape((5, 5))
    # b = ndimage.maximum_filter(b, footprint=np.ones((3, 3)))
    # z = (b > 5) & (b < 20)
    # print(a)
    # b[(b == 5) | (b == 10) | (b == 12)] = 0
    # print(np.gradient(a)[0], '\n', np.gradient(a)[1])
    # Tests.zero_crossing_check()
    # Tests.vectorize_check(5, 0.04, 0.1)
    Tests.my_edge_detection_check(0.975, 0.995)
    # Tests.laplacian_edge_detection_check(0.9725)
    # Tests.canny_detector_check(20, 7)
