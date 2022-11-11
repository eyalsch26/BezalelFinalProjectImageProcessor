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


if __name__ == '__main__':
    a = np.zeros((2, 5, 5))
    b = np.arange(25).reshape((5, 5))
    a[1][b > 4] = 1
    a[1] *= b
    print(a)
    # Tests.laplacian_edge_detection()
    # Tests.corner_detection_sobel_check()
