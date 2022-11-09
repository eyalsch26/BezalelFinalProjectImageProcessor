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
    a = np.arange(10)
    a[2 < a < 7] = 100
    print(a)
    # Tests.laplacian_edge_detection()
    # Tests.corner_detection_sobel_check()
