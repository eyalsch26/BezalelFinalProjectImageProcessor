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
    Renderer.render_content_setup()
    # Renderer.render_cubist(False, False, True)
    # FileManager.raster_contour_from_file(
    #     '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Jellyfish'
    #     '/Rasterization_Jellyfish_3.txt', os='m')
    # FileManager.volume_colourizer(
    #     '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Jellyfish'
    #     '/Colourization_Jellyfish_2.txt', os='m')

    # Tests.trace_edge_from_corner_basic_check()
    # Tests.save_rgba_check()
    # Tests.pixels_count()
    # Tests.vector_strokes_displacement_check(FileManager.FRAME_IN, FileManager.RAST_DIR_OUT)
    # Tests.vectorize_check()
    # Tests.vectorize_check_mac()
    # Tests.write_bzr_ctrl_pts_to_file_check()
    # Tests.read_bzr_ctrl_pts_from_file_check()
    # Tests.colour_image_check_mac()
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
    # Tests.volume_colourizer_check_mac()
