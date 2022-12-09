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
    # a = np.array([[[0.5, 0.001], [0, 0], [0, 0]], [[0, 0], [1.1, 1.9], [0, 0]], [[0, 0], [0, 0], [0, 0]]])
    # b = np.array([[[0, 0], [0, 0], [0, 0]], [[0, 0], [-1, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]])
    # c = np.array([a, b])
    # d = np.array([[2, 0, 0], [0, 5, 0], [1, 0, 3]])
    # e = np.zeros((3, 3))
    # print(e)
    # print(np.linalg.norm(c, axis=2))
    # Tests.vectorize_check()
    # Tests.show_reel('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Render Test\HD720\Aang_Sequence.',
    #                 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\SequenceTest\Aang_Sequence_Res.',
    #                 0, 130, 4)
    Tests.rasterizer_check()
    # Tests.my_edge_detection_check(0.975, 0.995)
