import numpy as np
import scipy
from scipy import ndimage
from scipy import sparse
from scipy import signal
import matplotlib.pyplot as plt
import math
import FileManager
import EdgeDetector
import Vectorizer
import Colourizer
import Rasterizer
import Tests


if __name__ == '__main__':
    # a = np.empty((0, 4, 2), np.float64)
    # r = 1.9
    # r_r = math.ceil(r)
    # x = np.linspace(-r_r, r_r, 2 * r_r + 1)
    # y = np.linspace(-r_r, r_r, 2 * r_r + 1)
    # xx, yy = np.meshgrid(x, y)
    # d = np.clip(r - np.sqrt(xx ** 2 + yy ** 2), 0, 1)
    # print(d)
    # c = np.ones((9, 16, 3))
    # c[::, ::, :1:] *= 3
    # c[::, ::, 1:2:] *= 4
    # c[::, ::, 2::] *= 5
    # print(c)
    # d = {1: np.array([[1, 1], [2, 1], [3, 1], [4, 1]]), 2: np.array([[2, 5], [3, 5], [6, 6]])}
    # z = map(np.ndarray.tolist, d.values())
    # c = np.append(a, [b], axis=0)
    # print(c)
    d = np.array([[[0, 1], [1, 2], [0, 1], [1, 2]],
                  [[0, 3], [1, 4], [0, 5], [1, 6]],
                  [[0, 7], [1, 8], [0, 9], [1, 10]]])
    e = np.average(d, axis=(0, 1))
    p = e - d
    print(e)
    print(p)
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
    # Tests.save_rgba_check()
    # Tests.pixels_count()
    # Tests.vector_strokes_displacement_check(FileManager.FRAME_IN, FileManager.RAST_DIR_OUT)
    # Tests.vectorize_check()
    # Tests.vectorize_check_mac()
    # Tests.write_bzr_ctrl_pts_to_file_check()
    # Tests.read_bzr_ctrl_pts_from_file_check()
    # Tests.colour_image_check()
    # Tests.displace_distort_colour_bcp_from_file_check()
    # Tests.displace_bcp_from_file_check()
    # Tests.displace_sequence_bcp_from_file_check(72)
    # Tests.displacement_check(FileManager.FRAME_IN, FileManager.VEC_DIR_OUT)
    # Tests.displacement_sequence_check_mac(FileManager.FRAME_IN_MAC,
    #                                       '/Users/eyalschaffer/Pictures/BezalelFinalProject/Output/Vector/Sequence', 24)
    # Tests.show_reel('G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\Render Test\HD720\Establish_00\Establish_00.',
    #                 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\SequenceTest\Establish_00\Establish_00_Res.',
    #                 0, 2276, 4)
    # Tests.sequence_face_stroke_rasterizer_check()
    # Tests.my_edge_detection_check(0.975, 0.995)
    # Tests.my_corner_detection_check(0.975, 0.995)
