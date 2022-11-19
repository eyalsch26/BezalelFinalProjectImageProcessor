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
    # a = np.array([[[0, 0], [0, 0], [0, 0]], [[0, 0], [1, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]])
    # b = np.array([[[0, 0], [0, 0], [0, 0]], [[0, 0], [-1, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]])
    # c = np.array([[0, 0, 0], [0, -np.sqrt(2), 0], [0, 0, 0]])
    # d = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    # print((np.dot(a[0], b[0]) / np.abs(a[0]) * np.abs(b[0])))
    # print(np.arctan2(a, b) * 180 / np.pi)
    # Tests.vectorize_check(5, 0.04, 0.1)
    # Tests.my_edge_detection_check(0.975, 0.995)
