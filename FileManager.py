import os
import numpy as np
from matplotlib import pyplot as plt
from imageio.v2 import imread
from matplotlib import image
import Colourizer


# Windows
FRAMES_DIR_IN = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input'
VEC_DIR_OUT = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Vector'
RAST_DIR_OUT = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Output\Raster\Stroke\SequenceFace'
FRAME_IN = 'G:\Eyal\Pictures\Bezalel\FinalProject\TestFrames\Input\\frame_0TryG2.png'
# Mac
FRAME_IN_MAC = '/Users/eyalschaffer/Documents/maya/projects/A_Moment_In_Life/images/Shape.01.png'
FRAMES_DIR_IN_MAC = '/Users/eyalschaffer/Documents/maya/projects/A_Moment_In_Life/images/Shape.01.png'
VEC_DIR_OUT_MAC = '/Users/eyalschaffer/Pictures/BezalelFinalProject/Output/Vector'
RAST_DIR_OUT_MAC = '/Users/eyalschaffer/Pictures/BezalelFinalProject/Output/Raster'


def import_image(path):
    frame = imread(path).astype(np.float64)
    if np.amax(frame) > 1:
        frame /= 255
    return frame


def save_image(dir_path, image_frame, frame_index, name_suffix, as_grayscale=True):
    # Naming the path of the new frame to be saved.
    path = dir_path + '\\frame_' + str(frame_index) + '_' + name_suffix + '.png'
    # Saving the path.
    if as_grayscale:
        image_frame = Colourizer.rgb_to_gray(image_frame)
        plt.imsave(path, image_frame, cmap='gray')
    else:
        plt.imsave(path, image_frame)
    return

