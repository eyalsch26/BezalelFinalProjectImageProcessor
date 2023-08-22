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
import Renderer
import Tests


if __name__ == '__main__':
    # Windows
    # FileManager.vectorize_contour_to_file('', os='w')
    # FileManager.raster_contour_from_file('', os='w')
    # Mac
    # Renderer.render_me()
    # Renderer.render_aang()
    # Renderer.render_face_test()
    Renderer.render_scans()
    # Renderer.render_logo()
    # Renderer.render_text(FileManager.RND_TXT_HEAD)
    # Renderer.render_background()
    # Renderer.render_creatures()
    # Renderer.render_content_setup()
    # Renderer.render_content_chase()
    # Renderer.render_content_Acquaintance()
    # Renderer.render_content_first_disassembly()
    # Renderer.render_content_first_disassembly_grays()
    # Renderer.render_content_second_disassembly()
    # Renderer.render_content_second_disassembly_solid()
    # Renderer.render_content_final_fusion()
    # Renderer.render_form_line_first_touch()
    # Renderer.render_form_line_first_disassembly()
    # Renderer.render_form_linear()
    # Renderer.render_content_cubist()
    # Renderer.render_form_cubist()
    # Renderer.render_form_smooth()
    # Renderer.render_butterfly_setup()
    # Renderer.render_butterfly_after_birth()
    # Renderer.render_butterfly_credits()
    # Renderer.render_butterfly_end()
    # Renderer.render_bubbles()
    # Renderer.render_square_frame()
    # Renderer.render_triangle_frame()
    # Renderer.render_rings_pyramid_frame()
    # Renderer.render_hollow_rock()
    # Renderer.render_content_smooth_phase()
    # Renderer.render_cubist(True, False, False)
    # FileManager.raster_contour_from_file(
    #     '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Jellyfish'
    #     '/Rasterization_Jellyfish_3.txt', os='m')
    # FileManager.volume_colourizer(
    #     '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Jellyfish'
    #     '/Colourization_Jellyfish_2.txt', os='m')

    # Tests.my_edge_detection_check()
    # Tests.my_corner_detection_check()
    # a = np.array([[0, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],  [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
    # b = np.zeros((8, 8))
    # b[a.T[0], a.T[1]] = 1
    # b = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 1, 1, 1, 1, 0, 0],
    #               [0, 0, 1, 1, 0, 1, 1, 0, 0],
    #               [0, 0, 1, 0, 1, 0, 0, 1, 1],
    #               [0, 0, 1, 1, 0, 1, 0, 1, 0],
    #               [0, 0, 0, 1, 1, 1, 0, 1, 0],
    #               [0, 0, 0, 0, 1, 1, 1, 0, 0],
    #               [1, 0, 0, 0, 0, 0, 0, 0, 0]])
    # c = Vectorizer.connectivity_component(b, [1, 2])
    # d = np.argwhere(b == 1)
    # print(c)
    # err = Vectorizer.calculate_path_curve_error_new(a, b)
    # Tests.trace_edges_check()
    # Tests.vectorize_check_mac()
