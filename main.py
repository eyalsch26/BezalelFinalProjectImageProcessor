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
    d = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    e = np.zeros((7, 7))
    e[2:5, 2:5] = d
    # f = np.unravel_index(np.argmax(e[2:5, 2:5]), e.shape)
    print(e)
    # print(f)
    # Tests.vectorize_check()
    # Tests.show_reel('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Render Test\HD720\Aang_Sequence.',
    #                 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\SequenceTest\Aang_Sequence_Res.',
    #                 0, 130, 4)
    # Tests.rasterizer_check()
    # Tests.my_edge_detection_check(0.975, 0.995)
