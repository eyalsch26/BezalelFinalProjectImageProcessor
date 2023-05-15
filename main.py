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
    # FileManager.vectorize_contour_to_file('G:\Eyal\Documents\Bezalel\FinalProject\ParametersFiles\ParametersFileVectorization_FormPOC.txt')
    FileManager.raster_contour_from_file('G:\Eyal\Documents\Bezalel\FinalProject\ParametersFiles'
                                  '\ParametersFileRasterization_FormPOC.txt')
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
