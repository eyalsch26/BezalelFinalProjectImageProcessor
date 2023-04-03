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
    # a = np.empty((0, 4, 2), np.float64)
    # b = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
    # c = np.append(a, [b], axis=0)
    # print(c)
    # d = np.array([[[0.0, 1.0], [1, 2.2]], [[0, 0], [1, 0]]])
    # c = np.zeros((np.max(d) + 1, np.max(d) + 1))
    # d_flatten = d[0].T[0] * 2 + d[0].T[1]
    # np.put(c, d_flatten, 1)
    # print(d[0].astype(np.uint32))
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
    # print(np.ceil(0.5))
    # e = np.ones((2 * r - 1, 2 * r - 1))
    # for row in range(r - 1):
    #     for column in range(r - 1):
    #         dist = np.round(np.sqrt((r - row) ** 2 + (r - column) ** 2))
    #         if dist > r:
    #             e[row][column] = 0
    # e *= e[::-1]
    # e *= e[::, ::-1]
    # f = np.unravel_index(np.argmax(e[2:5, 2:5]), e.shape)
    # Tests.trace_edge_from_corner_basic_check()
    # Tests.vectorize_check()
    # Tests.vectorize_check_mac()
    Tests.displacement_sequence_check_mac(FileManager.FRAME_IN_MAC,
                                          '/Users/eyalschaffer/Pictures/BezalelFinalProject/Output/Vector/Sequence', 24)
    # Tests.show_reel('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Render Test\HD720\Establish_00\Establish_00.',
    #                 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\SequenceTest\Establish_00\Establish_00_Res.',
    #                 0, 2276, 4)
    # Tests.sequence_face_stroke_rasterizer_check()
    # Tests.my_edge_detection_check(0.975, 0.995)
