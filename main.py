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
    Renderer.render_background()
    # Renderer.render_content_setup()
    # Renderer.render_content_cubist()
    # Renderer.render_form_cubist()
    # Renderer.render_cubist(True, False, False)
    # FileManager.raster_contour_from_file(
    #     '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Jellyfish'
    #     '/Rasterization_Jellyfish_3.txt', os='m')
    # FileManager.volume_colourizer(
    #     '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Jellyfish'
    #     '/Colourization_Jellyfish_2.txt', os='m')
