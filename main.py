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
    # d = np.array([[0, 0, 1, 1, 0],
    #               [0, 1, 0, 0, 0],
    #               [0, 1, 0, 0, 0],
    #               [0, 1, 0, 1, 0],
    #               [0, 0, 1, 0, 0]])
    # r = np.argwhere(d == 1)
    # h = np.min(r, axis=0)
    # q = np.max(r, axis=0)
    # print(r)
    # print(h)
    # print(q)
    print(np.ceil(0.5))
    # e = np.ones((2 * r - 1, 2 * r - 1))
    # for row in range(r - 1):
    #     for column in range(r - 1):
    #         dist = np.round(np.sqrt((r - row) ** 2 + (r - column) ** 2))
    #         if dist > r:
    #             e[row][column] = 0
    # e *= e[::-1]
    # e *= e[::, ::-1]
    # f = np.unravel_index(np.argmax(e[2:5, 2:5]), e.shape)
    # Tests.vectorize_check()
    # Tests.show_reel('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Render Test\HD720\Aang_Sequence.',
    #                 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\SequenceTest\Aang_Sequence_Res.',
    #                 0, 130, 4)
    # Tests.sequence_face_stroke_rasterizer_check()
    # Tests.my_edge_detection_check(0.975, 0.995)
