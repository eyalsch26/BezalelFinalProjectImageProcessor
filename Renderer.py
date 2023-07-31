import numpy as np
from matplotlib import pyplot as plt
from imageio.v2 import imread
from matplotlib import image
import FileManager
import Colourizer
import Rasterizer
import Vectorizer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ General ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def volume_colourizer(parameters_path, os='w'):
    dir_in_path, f_prefix, out_path, start, end, digits_num, alpha_t, c, blr_k, tlr, clr_s, rgb_from_im, r, g, b = \
        FileManager.import_parameters(parameters_path)
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the output file name.
        n_padded = f'{im_file_idx}'
        while (len(n_padded) < digits_num):
            n_padded = f'0{n_padded}'
        # Preparing the image and the filter.
        im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        im = FileManager.import_image(im_name)
        if rgb_from_im == 'True':
            r = np.max(im[::, ::, 0])
            g = np.max(im[::, ::, 1])
            b = np.max(im[::, ::, 2])
        a = Vectorizer.blur_image(Colourizer.alpha_channel(im[::, ::, 3], alpha_t, c), int(blr_k))
        shape = im[:, :, 0].shape
        txr_im = Rasterizer.texture_frame(shape)
        r_im = txr_im * (r / 255)
        g_im = txr_im * (g / 255)
        b_im = txr_im * (b / 255)
        im_rgb = np.clip(np.dstack((r_im, g_im, b_im)), 0, 1)
        clr_im_name = f'{f_prefix}.{n_padded}'
        FileManager.save_rgba_image(out_path, clr_im_name, im_rgb, a, os)


# Vectorization
def vectorize_contour_to_file(parameters_path, os='w'):
    dir_in_path, f_prefix, dir_out_bcp_path, start, end, digits_num, min_crv_ratio = FileManager.import_parameters(parameters_path)
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        # Preparing the image.
        im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        im = FileManager.import_image(im_name)
        im_yiq = Colourizer.rgb_to_yiq(im)
        im_y = im_yiq[:, :, 0]
        # Finding the Bezier control points.
        im_bcp = Vectorizer.vectorize_image_new(im_y, min_crv_ratio)
        # Saving the Bezier control points to a file.
        bcp_f_name = FileManager.file_path(dir_out_bcp_path, f_prefix, n_padded, 'txt', os)
        FileManager.save_bezier_control_points(bcp_f_name, im_bcp)


def vectorize_text_contour_to_file(parameters_path, os='w'):
    dir_in_path, f_prefix, dir_out_bcp_path, start, end, digits_num, min_crv_ratio = FileManager.import_parameters(parameters_path)
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        # Preparing the image.
        im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        im = FileManager.import_image(im_name)
        im_yiq = Colourizer.rgb_to_yiq(im)
        im_y = im_yiq[:, :, 0]
        # Finding the Bezier control points.
        im_bcp = Vectorizer.vectorize_image_new(im_y, min_crv_ratio)
        # Saving the Bezier control points to a file.
        bcp_f_name = FileManager.file_path(dir_out_bcp_path, f_prefix, n_padded, 'txt', os, True)
        FileManager.save_bezier_control_points(bcp_f_name, im_bcp)


# Not in use. Content is rasterized directly.
def vectorize_content_to_file(parameters_path, os='w'):
    dir_in_path, f_prefix, dir_out_bcp_path, start, end, digits_num, min_crv_ratio = FileManager.import_parameters(parameters_path)
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        # Preparing the image.
        im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        im = FileManager.import_image(im_name)
        im_yiq = Colourizer.rgb_to_yiq(im)
        im_y = im_yiq[:, :, 0]
        # Finding the Bezier control points.
        im_bcp = Vectorizer.vectorize_content_image(im_y, min_crv_ratio)
        # Saving the Bezier control points to a file.
        bcp_f_name = FileManager.file_path(dir_out_bcp_path, f_prefix, n_padded, 'txt', os)
        FileManager.save_bezier_control_points(bcp_f_name, im_bcp)


# Rasterization
def raster_contour_from_file(parameters_path, os='w'):
    dir_in_path, f_prefix, dir_out_path, start, end, digits_num, canvas_input_x, canvas_input_y, canvas_output_x,\
    canvas_output_y, scale, scale_factor, contiguous, diminish, diminish_min_l_r, dsp_dst_direction, dsp_dst_style, \
    displace, displace_min, displace_max, displace_transform_max, distort, distort_min, distort_max, strk_w_min, \
    strk_w_max, texture_style, texture_type, colour_style, r_min, r_max, g_min, g_max, b_min, b_max, alpha, alpha_c, \
    alpha_f = FileManager.import_parameters(parameters_path)
    cnvs_shape = (int(canvas_output_x), int(canvas_output_y))
    canvas_scaler = canvas_output_x / canvas_input_x
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        im_name = f'{f_prefix}.{n_padded}'
        # Preparing the image.
        im_r = np.zeros(cnvs_shape)
        im_g = np.zeros(cnvs_shape)
        im_b = np.zeros(cnvs_shape)
        im_a = np.zeros(cnvs_shape)
        # Importing the Bezier control points from the text file.
        bcp_f_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'txt', os)
        bcp_arr, bcp_num = FileManager.import_bezier_control_points(bcp_f_name)
        # Checking if there are any Bezier control points to raster.
        if bcp_num < 1:
            im_rgb = np.dstack((im_r, im_g, im_b))
            FileManager.save_rgba_image(dir_out_path, im_name, im_rgb, im_a, os)
            continue
        bcp_arr *= canvas_scaler
        # Applying vector manipulation.
        idx_factor = Vectorizer.index_displace_distort_factor(im_file_idx, start, end, dsp_dst_direction, dsp_dst_style)
        if diminish == 'True':
            bcp_arr = Rasterizer.diminish_bcps_num(bcp_arr, float(diminish_min_l_r))
        if scale == 'True':
            s_f = idx_factor ** scale_factor
            bcp_arr = Vectorizer.scale_bezier_curves(bcp_arr, s_f)
        if displace == 'True':
            dsp_f = int(displace_min + idx_factor * (displace_max - displace_min))
            dsp_t = int(idx_factor * displace_transform_max)
            bcp_arr = Vectorizer.displace_bezier_curves(bcp_arr, dsp_f, dsp_t)
        if distort == 'True':
            dst_f = int(distort_min + idx_factor * (distort_max - distort_min))
            bcp_arr = Vectorizer.distort_bezier_curves(bcp_arr, dst_f)
        # Rastering the curves.
        curves_num = len(bcp_arr)
        txr_arr = Rasterizer.generate_textures_arr(curves_num, texture_style, texture_type)  # Defining texture.
        rgb_range = np.array([[r_min, r_max], [g_min, g_max], [b_min, b_max]], dtype=int)
        clr_arr = Colourizer.generate_colours_arr(curves_num, colour_style, rgb_range)  # Defining colour.
        for crv_idx in range(curves_num):
            cur_bcp = bcp_arr[crv_idx]
            cur_txr = txr_arr[crv_idx]
            cur_clr = clr_arr[crv_idx]
            stroke = Rasterizer.stroke_rasterizer(cur_bcp, strk_w_min, strk_w_max, texture=cur_txr,
                                                   canvas_shape=cnvs_shape)
            stroke_rgb = np.repeat(stroke, Colourizer.CLR_DIM).reshape((int(canvas_output_x), int(canvas_output_y),
                                                                        Colourizer.CLR_DIM))
            stroke_rgb = Colourizer.colour_stroke(stroke_rgb, cur_clr[0], cur_clr[1], cur_clr[2])
            stroke_alpha_im = Colourizer.alpha_channel(stroke, alpha, alpha_c, int(alpha_f))
            stroke_im_r = stroke_rgb[::, ::, :1:].reshape(cnvs_shape)
            stroke_im_g = stroke_rgb[::, ::, 1:2:].reshape(cnvs_shape)
            stroke_im_b = stroke_rgb[::, ::, 2::].reshape(cnvs_shape)
            im_r = Colourizer.composite_rgb(stroke_im_r, im_r, stroke_alpha_im)
            im_g = Colourizer.composite_rgb(stroke_im_g, im_g, stroke_alpha_im)
            im_b = Colourizer.composite_rgb(stroke_im_b, im_b, stroke_alpha_im)
            im_a = Colourizer.composite_alpha(stroke_alpha_im, im_a)
        im_rgb = np.dstack((im_r, im_g, im_b))
        FileManager.save_rgba_image(dir_out_path, im_name, im_rgb, im_a, os)


def raster_content_from_file(parameters_path, os='w'):
    dir_in_path, f_prefix, dir_out_path, start, end, digits_num, canvas_input_x, canvas_input_y, canvas_output_x,\
    canvas_output_y, scale, scale_factor, contiguous, diminish, diminish_coefficient, dsp_dst_direction, \
    dsp_dst_style, displace, displace_min, displace_max, displace_transform_max, distort, distort_min, distort_max, \
    strk_w_min, strk_w_max, texture_style, texture_type, colour_style, r_min, r_max, g_min, g_max, b_min, b_max, \
    alpha, alpha_c, alpha_f = FileManager.import_parameters(parameters_path)
    cnvs_shape = (int(canvas_output_x), int(canvas_output_y))
    canvas_scaler = canvas_output_x / canvas_input_x
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        im_id = f'{f_prefix}.{n_padded}'
        # Preparing the image.
        im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        im = FileManager.import_image(im_name)
        idx_factor = 1
        if scale == 'True':
            idx_factor = Vectorizer.index_displace_distort_factor(im_file_idx, start, end, dsp_dst_direction, dsp_dst_style)
        # Rastering the image.
        im_rgb, im_a = Rasterizer.content_rasterizer(im, cnvs_shape, canvas_scaler, idx_factor, diminish,
                                                     diminish_coefficient, displace, displace_transform_max,
                                                     strk_w_min, 3, texture_style, texture_type, r_min, r_max, g_min,
                                                     g_max, b_min, b_max, colour_style, alpha, alpha_c, alpha_f)
        FileManager.save_rgba_image(dir_out_path, im_id, im_rgb, im_a, os)
    return


def raster_content_smooth_from_file(parameters_path, os='w'):
    dir_in_path, f_prefix, dir_out_path, start, end, digits_num = FileManager.import_parameters(parameters_path)
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        im_id = f'{f_prefix}.{n_padded}'
        # Preparing the image.
        im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        im = FileManager.import_image(im_name)
        # Rastering the image.
        im_rgb, im_a = Rasterizer.content_smooth_rasterizer(im)
        FileManager.save_rgba_image(dir_out_path, im_id, im_rgb, im_a, os)
    return


def render_creature_contour(parameters_path, os='w'):
    dir_in_path, f_prefix, dir_out_path, start, end, digits_num, canvas_input_x, canvas_input_y, canvas_output_x,\
    canvas_output_y, scale, scale_factor, contiguous, diminish, diminish_coefficient, dsp_dst_direction, \
    dsp_dst_style, displace, displace_min, displace_max, displace_transform_max, distort, distort_min, distort_max, \
    strk_w_min, strk_w_max, texture_style, texture_type, colour_style, r_min, r_max, g_min, g_max, b_min, b_max, \
    alpha, alpha_c, alpha_f = FileManager.import_parameters(parameters_path)
    cnvs_shape = (int(canvas_output_x), int(canvas_output_y))
    canvas_scaler = canvas_output_x / canvas_input_x
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        im_id = f'{f_prefix}.{n_padded}'
        # Preparing the image.
        im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        im = FileManager.import_image(im_name)
        idx_factor = 1
        if scale == 'True':
            idx_factor = Vectorizer.index_displace_distort_factor(im_file_idx, start, end, dsp_dst_direction, dsp_dst_style)
        # Rastering the image.
        im_rgb, im_a = Rasterizer.creature_contour_rasterizer(im, cnvs_shape, canvas_scaler, idx_factor, diminish,
                                                     diminish_coefficient, displace, displace_transform_max,
                                                     strk_w_min, 3, texture_style, texture_type, r_min, r_max, g_min,
                                                     g_max, b_min, b_max, colour_style, alpha, alpha_c, alpha_f)
        FileManager.save_rgba_image(dir_out_path, im_id, im_rgb, im_a, os)


def raster_text_contour_from_file(parameters_path, os='w'):
    dir_in_path, f_prefix, dir_out_path, start, end, digits_num, canvas_input_x, canvas_input_y, canvas_output_x,\
    canvas_output_y, scale, scale_factor, contiguous, diminish, diminish_min_l_r, dsp_dst_direction, dsp_dst_style, \
    displace, displace_min, displace_max, displace_transform_max, distort, distort_min, distort_max, strk_w_min, \
    strk_w_max, texture_style, texture_type, colour_style, r_min, r_max, g_min, g_max, b_min, b_max, alpha, alpha_c, \
    alpha_f = FileManager.import_parameters(parameters_path)
    cnvs_shape = (int(canvas_output_x), int(canvas_output_y))
    canvas_scaler = canvas_output_x / canvas_input_x
    # Importing the Bezier control points from the text file.
    bcp_f_name = FileManager.file_path(dir_in_path, f_prefix, 0, 'txt', os, True)
    org_bcp_arr, bcp_num = FileManager.import_bezier_control_points(bcp_f_name)
    org_bcp_arr *= canvas_scaler
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        bcp_arr = org_bcp_arr
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        im_name = f'{f_prefix}.{n_padded}'
        # Preparing the image.
        im_r = np.zeros(cnvs_shape)
        im_g = np.zeros(cnvs_shape)
        im_b = np.zeros(cnvs_shape)
        im_a = np.zeros(cnvs_shape)
        # Checking if there are any Bezier control points to raster.
        if bcp_num < 1:
            continue
        # Applying vector manipulation.
        idx_factor = Vectorizer.index_displace_distort_factor(im_file_idx, start, end, dsp_dst_direction, dsp_dst_style)
        if diminish == 'True':
            bcp_arr = Rasterizer.diminish_bcps_num(bcp_arr, float(diminish_min_l_r))
        if scale == 'True':
            s_f = scale_factor * idx_factor
            bcp_arr = Vectorizer.scale_bezier_curves(bcp_arr, s_f)
        if displace == 'True':
            dsp_f = int(displace_min + idx_factor * (displace_max - displace_min))
            dsp_t = int(idx_factor * displace_transform_max)
            bcp_arr = Vectorizer.displace_bezier_curves(bcp_arr, dsp_f, dsp_t)
        if distort == 'True':
            dst_f = int(distort_min + idx_factor * (distort_max - distort_min))
            bcp_arr = Vectorizer.distort_bezier_curves(bcp_arr, dst_f)
        # Rastering the curves.
        curves_num = len(bcp_arr)
        txr_arr = Rasterizer.generate_textures_arr(curves_num, texture_style, texture_type)  # Defining texture.
        rgb_range = np.array([[r_min, r_max], [g_min, g_max], [b_min, b_max]], dtype=int)
        clr_arr = Colourizer.generate_colours_arr(curves_num, colour_style, rgb_range)  # Defining colour.
        for crv_idx in range(curves_num):
            cur_bcp = bcp_arr[crv_idx]
            cur_txr = txr_arr[crv_idx]
            cur_clr = clr_arr[crv_idx]
            stroke = Rasterizer.stroke_rasterizer(cur_bcp, strk_w_min, strk_w_max, texture=cur_txr, canvas_shape=cnvs_shape)
            stroke_rgb = np.repeat(stroke, Colourizer.CLR_DIM).reshape((int(canvas_output_x), int(canvas_output_y),
                                                                        Colourizer.CLR_DIM))
            stroke_rgb = Colourizer.colour_stroke(stroke_rgb, cur_clr[0], cur_clr[1], cur_clr[2])
            alpha_c = np.clip((1 - idx_factor) * alpha_c + np.random.randint(0, end - start + 1) / (end - start + 1),
                              0, 1)
            stroke_alpha_im = Colourizer.alpha_channel(stroke, alpha, alpha_c, int(alpha_f))
            stroke_im_r = stroke_rgb[::, ::, :1:].reshape(cnvs_shape)
            stroke_im_g = stroke_rgb[::, ::, 1:2:].reshape(cnvs_shape)
            stroke_im_b = stroke_rgb[::, ::, 2::].reshape(cnvs_shape)
            im_r = Colourizer.composite_rgb(stroke_im_r, im_r, stroke_alpha_im)
            im_g = Colourizer.composite_rgb(stroke_im_g, im_g, stroke_alpha_im)
            im_b = Colourizer.composite_rgb(stroke_im_b, im_b, stroke_alpha_im)
            im_a = Colourizer.composite_alpha(stroke_alpha_im, im_a)
        im_rgb = np.dstack((im_r, im_g, im_b))
        FileManager.save_rgba_image(dir_out_path, im_name, im_rgb, im_a, os)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Text ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def render_text(parameters_path):
    vectorize, rasterize, colourize, vectorize_path, rasterize_path, colourize_path, os = FileManager.import_parameters(
        parameters_path)
    if vectorize == 'True':
        vectorize_text_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_text_contour_from_file(rasterize_path, os)
    return


def render_letters():
    vectorize, rasterize, vectorize_path, general_rasterize_path, os = FileManager.import_parameters(
        FileManager.RND_TXT_LETTERS)
    abc = 'abcdefghijklmnopqrstuvwxyz'
    ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for letter_idx in range(len(abc)):
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            rasterize_path = general_rasterize_path + abc[letter_idx]
            raster_text_contour_from_file(rasterize_path, os)
    return


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Background ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def render_background_raster(parameters_path, os):
    dir_in_path, f_prefix, dir_out_path, start, end, digits_num, canvas_input_x, canvas_input_y, canvas_output_x, \
    canvas_output_y = FileManager.import_parameters(parameters_path)
    cnvs_shape = (int(canvas_output_x), int(canvas_output_y))
    im_r = np.ones(cnvs_shape)
    im_g = np.ones(cnvs_shape)
    im_b = np.ones(cnvs_shape)
    im_rgb = np.dstack((im_r, im_g, im_b))
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input/output file name.
        n_padded = f'{im_file_idx}'
        while len(n_padded) < digits_num:
            n_padded = f'0{n_padded}'
        im_id = f'{f_prefix}.{n_padded}'
        # Preparing the image.
        # im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        # im = FileManager.import_image(im_name)
        # im_rgb = im[:, :, :3]
        # Rastering the image.
        im_a = Rasterizer.background_rasterizer(cnvs_shape)
        FileManager.save_rgba_image(dir_out_path, im_id, im_rgb, im_a, os)
    return


def render_background_interpolation(parameters_path, os):
    dir_in_path, f_prefix, dir_out_path, start, end, digits_num, canvas_input_x, canvas_input_y, canvas_output_x, \
    canvas_output_y, interpolation_num = FileManager.import_parameters(parameters_path)
    cnvs_shape = (int(canvas_output_x), int(canvas_output_y))
    im_r = np.ones(cnvs_shape)
    im_g = np.ones(cnvs_shape)
    im_b = np.ones(cnvs_shape)
    im_rgb = np.dstack((im_r, im_g, im_b))
    # Iterating over the desired images.
    for im_file_idx in range(int(start), int(end) + 1):
        # Preparing the input file name.
        n_padded_start = f'{im_file_idx}'
        n_padded_end = f'{im_file_idx + 1}'
        if im_file_idx == end:  # Cyclic interpolation between the last and first frames.
            n_padded_end = f'{int(start)}'
        while len(n_padded_start) < digits_num:
            n_padded_start = f'0{n_padded_start}'
        while len(n_padded_end) < digits_num:
            n_padded_end = f'0{n_padded_end}'
        # Preparing the image.
        im_name_start = FileManager.file_path(dir_in_path, f_prefix, n_padded_start, 'png', os)
        im_name_end = FileManager.file_path(dir_in_path, f_prefix, n_padded_end, 'png', os)
        im_start = FileManager.import_image(im_name_start)
        im_end = FileManager.import_image(im_name_end)
        im_a_start = im_start[:, :, 3]
        im_a_end = im_end[:, :, 3]
        # Interpolating the image.
        for idx in range(int(interpolation_num)):
            idx_factor = idx / (interpolation_num - 1)
            im_a = Rasterizer.background_interpolation(im_a_start, im_a_end, idx_factor)
            # Preparing the output file name.
            im_id_padded = f'{(im_file_idx - 1) * int(interpolation_num) + idx + 1}'
            while len(im_id_padded) < digits_num:
                im_id_padded = f'0{im_id_padded}'
            im_id = f'{f_prefix}.{im_id_padded}'
            FileManager.save_rgba_image(dir_out_path, im_id, im_rgb, im_a, os)
    return


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Creatures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def render_Jellyfish(parameters_path):
    vectorize, rasterize, colourize, vectorize_path, rasterize_path, colourize_path = FileManager.import_parameters(
        parameters_path)
    raster_contour_from_file(
        '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Jellyfish'
        '/Rasterization_Jellyfish_3.txt', os='m')
    volume_colourizer(
        '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Jellyfish'
        '/Colourization_Jellyfish_2.txt', os='m')
    return


def render_butterfly_setup():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(FileManager.RND_BTRFY_SETUP)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_butterfly_after_birth():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(FileManager.RND_BTRFY_AFTRBRTH)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_butterfly_credits():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
        FileManager.RND_BTRFY_CRDTS)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_butterfly_end():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
        FileManager.RND_BTRFY_END)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_bubbles():
    bubble_0, bubble_1, bubble_2, bubble_3, bubble_4, bubble_5 = False, False, True, True, True, True
    if bubble_0:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_BUBBLE_0)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if bubble_1:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_BUBBLE_1)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if bubble_2:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_BUBBLE_2)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if bubble_3:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_BUBBLE_3)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if bubble_4:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_BUBBLE_4)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if bubble_5:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_BUBBLE_5)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    return


def render_square_frame():
    low_angle, high_angle, front, back = True, True, True, True
    if low_angle:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_SQRE_FRAME_LOW)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if high_angle:
        if front:
            vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
                FileManager.RND_SQRE_FRAME_HIGH_FRNT)
            if vectorize == 'True':
                vectorize_contour_to_file(vectorize_path, os)
            if rasterize == 'True':
                raster_contour_from_file(rasterize_path, os)
        if back:
            vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
                FileManager.RND_SQRE_FRAME_HIGH_BACK)
            if vectorize == 'True':
                vectorize_contour_to_file(vectorize_path, os)
            if rasterize == 'True':
                raster_contour_from_file(rasterize_path, os)
    return


def render_triangle_frame():
    avoid, chase = True, True
    if avoid:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_TRI_FRAME_AVOID)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if chase:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_TRI_FRAME_CHASE)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    return


def render_rings_pyramid_frame():
    complete, partial = False, True
    if complete:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_RNGS_PRMD_CMPLT_FRONT)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_RNGS_PRMD_CMPLT_BACK)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if partial:
        # vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
        #     FileManager.RND_RNGS_PRMD_PRT_FRONT)
        # if vectorize == 'True':
        #     vectorize_contour_to_file(vectorize_path, os)
        # if rasterize == 'True':
        #     raster_contour_from_file(rasterize_path, os)
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_RNGS_PRMD_PRT_BACK)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    return


def render_hollow_rock():
    acquaintance, crdts = True, True
    if acquaintance:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_HOLLOW_ROCK_ACQ)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    if crdts:
        vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
            FileManager.RND_HOLLOW_ROCK_CRDTS)
        if vectorize == 'True':
            vectorize_contour_to_file(vectorize_path, os)
        if rasterize == 'True':
            raster_contour_from_file(rasterize_path, os)
    return


def render_face_test():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(FileManager.RND_FACE)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_aang():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(FileManager.RND_AANG)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_me():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(FileManager.RND_ME)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


# ---------------------------------------------------- Content ---------------------------------------------------------
def render_content(parameters_path):
    convergence, stable, divergence, convergence_rasterize_path, stable_rasterize_path, divergence_rasterize_path, \
    os = FileManager.import_parameters(
        parameters_path)
    if convergence == 'True':
        raster_content_from_file(convergence_rasterize_path, os)
    if stable == 'True':
        raster_content_from_file(stable_rasterize_path, os)
    if divergence == 'True':
        raster_content_from_file(divergence_rasterize_path, os)
    return


def render_content_cubist_phase(parameters_path):
    raster_front_left, raster_front_right, raster_rear_left, raster_rear_right, raster_front_left_path, \
    raster_front_right_path, raster_rear_left_path, raster_rear_right_path, os = FileManager.import_parameters(
    parameters_path)
    if raster_front_left == 'True':
        raster_content_from_file(raster_front_left_path, os)
    if raster_front_right == 'True':
        raster_content_from_file(raster_front_right_path, os)
    if raster_rear_left == 'True':
        raster_content_from_file(raster_rear_left_path, os)
    if raster_rear_right == 'True':
        raster_content_from_file(raster_rear_right_path, os)
    return


def render_content_smooth_phase():
    os = 'm'
    raster_content_smooth_from_file(FileManager.RND_CNT_SMOOTH_BIRTH, os)
    raster_content_smooth_from_file(FileManager.RND_CNT_SMOOTH_SNIF, os)
    raster_content_smooth_from_file(FileManager.RND_CNT_SMOOTH_WIPE, os)
    raster_content_smooth_from_file(FileManager.RND_CNT_SMOOTH_GALLOP, os)
    raster_content_smooth_from_file(FileManager.RND_CNT_SMOOTH_JUMP, os)
    raster_content_smooth_from_file(FileManager.RND_CNT_SMOOTH_SIT, os)
    return


# ----------------------------------------------------- Cubist ---------------------------------------------------------
def vectorize_cubist_form_to_file(parameters_path):
    vectorize_front_left, vectorize_front_right, vectorize_rear_left, vectorize_rear_right, \
    vectorize_front_left_path, vectorize_front_right_path, vectorize_rear_left_path, vectorize_rear_right_path, \
    os = FileManager.import_parameters(parameters_path)
    if vectorize_front_left == 'True':
        vectorize_contour_to_file(vectorize_front_left_path, os)
    if vectorize_front_right == 'True':
        vectorize_contour_to_file(vectorize_front_right_path, os)
    if vectorize_rear_left == 'True':
        vectorize_contour_to_file(vectorize_rear_left_path, os)
    if vectorize_rear_right == 'True':
        vectorize_contour_to_file(vectorize_rear_right_path, os)
    return


def raster_cubist_form_to_file(parameters_path):
    raster_front_left, raster_front_right, raster_rear_left, raster_rear_right, \
    raster_front_left_convergence_path, raster_front_left_stable_path, raster_front_left_divergence_path, \
    raster_front_right_convergence_path, raster_front_right_stable_path, raster_front_right_divergence_path, \
    raster_rear_left_convergence_path, raster_rear_left_stable_path, raster_rear_left_divergence_path, \
    raster_rear_right_convergence_path, raster_rear_right_stable_path, raster_rear_right_divergence_path, \
    os = FileManager.import_parameters(parameters_path)
    if raster_front_left == 'True':
        raster_contour_from_file(raster_front_left_convergence_path, os)
        raster_contour_from_file(raster_front_left_stable_path, os)
        raster_contour_from_file(raster_front_left_divergence_path, os)
    if raster_front_right == 'True':
        raster_contour_from_file(raster_front_right_convergence_path, os)
        raster_contour_from_file(raster_front_right_stable_path, os)
        raster_contour_from_file(raster_front_right_divergence_path, os)
    if raster_rear_left == 'True':
        raster_contour_from_file(raster_rear_left_convergence_path, os)
        raster_contour_from_file(raster_rear_left_stable_path, os)
        raster_contour_from_file(raster_rear_left_divergence_path, os)
    if raster_rear_right == 'True':
        raster_contour_from_file(raster_rear_right_convergence_path, os)
        raster_contour_from_file(raster_rear_right_stable_path, os)
        raster_contour_from_file(raster_rear_right_divergence_path, os)
    return


def render_cubist(vectorize, rasterize, colourize):
    if vectorize:
        vectorize_contour_to_file(
            '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
            '/Vectorization_Cubist_FrontLeft.txt', os='m')
        vectorize_contour_to_file(
            '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
            '/Vectorization_Cubist_FrontRight.txt', os='m')
        vectorize_contour_to_file(
            '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
            '/Vectorization_Cubist_RearLeft.txt', os='m')
        vectorize_contour_to_file(
            '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
            '/Vectorization_Cubist_RearRight.txt', os='m')
    if rasterize:
        for i in range(3):
            raster_contour_from_file(
                f'/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
                f'/Rasterization_Cubist_FrontLeft_{i}.txt',
                os='m')
        for i in range(3):
            raster_contour_from_file(
                f'/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
                f'/Rasterization_Cubist_FrontRight_{i}.txt', os='m')
        for i in range(3):
            raster_contour_from_file(
                f'/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
                f'/Rasterization_Cubist_RearLeft_{i}.txt', os='m')
        for i in range(3):
            raster_contour_from_file(
                f'/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
                f'/Rasterization_Cubist_RearRight_{i}.txt', os='m')
    if colourize:
        volume_colourizer(
            '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
            '/Colourization_Cubist_FrontLeft.txt', os='m')
        volume_colourizer(
            '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
            '/Colourization_Cubist_FrontRight.txt', os='m')
        volume_colourizer(
            '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
            '/Colourization_Cubist_RearLeft.txt', os='m')
        volume_colourizer(
            '/Users/eyalschaffer/Documents/Bezalel/FinalProject/DataFiles/ParametersFiles/Form/Cubist'
            '/Colourization_Cubist_RearRight.txt', os='m')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Commands ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def render_headline():
    render_text(FileManager.RND_TXT_HEAD)
    return


def render_logo():
    render_text(FileManager.RND_LOGO)


def render_background():
    raster, interpolate, raster_path, interpolation_path, os = FileManager.import_parameters(FileManager.RND_BG)
    if raster == 'True':
        render_background_raster(raster_path, os)
    if interpolate == 'True':
        render_background_interpolation(interpolation_path, os)
    return


def render_creature(parameters_path):
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
        parameters_path)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return
    # render_creature_contour()
    # setup, after_birth, at_credits, end, setup_path, after_birth_path, at_credits_path, end_path, \
    # os = FileManager.import_parameters(parameters_path)
    # if rasterize_contour == 'True':
    #     render_creature_contour(rasterize_contour_path)
    # if rasterize_volume == 'True':
    #     volume_colourizer(rasterize_volume_path)
    # if setup == 'True':
    #     cre



def render_content_setup():
    render_content(FileManager.RND_CNT_SETUP)
    return


def render_content_chase():
    render_content(FileManager.RND_CNT_CHASE)
    return


def render_content_Acquaintance():
    render_content(FileManager.RND_CNT_ACQTNCE)
    return


def render_content_first_disassembly():
    render_content(FileManager.RND_CNT_FIRST_DSASMBLY)
    return


def render_content_first_disassembly_grays():
    render_content(FileManager.RND_CNT_FIRST_DSASMBLY_GRY)
    return


def render_content_second_disassembly():
    render_content(FileManager.RND_CNT_SECOND_DSASMBLY)
    return


def render_content_second_disassembly_solid():
    render_content(FileManager.RND_CNT_SECOND_DSASMBLY_SLD)
    return


def render_content_final_fusion():
    render_content(FileManager.RND_CNT_FINL_FUSN)
    return


def render_form_line_first_touch():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
        FileManager.RND_FRM_LINE_FRST_TCH)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_form_line_first_disassembly():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(
        FileManager.RND_FRM_LINE_FRST_DSASMBLY)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    # render_content(FileManager.RND_FRM_LINE_FRST_DSASMBLY)
    return


def render_form_linear():
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(FileManager.RND_FRM_LNR)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_content_cubist():
    convergence, stable, divergence = FileManager.import_parameters(FileManager.RND_CNT_CUBST)
    if convergence == 'True':
        render_content_cubist_phase(FileManager.RND_CNT_CUBST_CONV)
    if stable == 'True':
        render_content_cubist_phase(FileManager.RND_CNT_CUBST_STBL)
    if divergence == 'True':
        render_content_cubist_phase(FileManager.RND_CNT_CUBST_DVRG)
    return


def render_form_cubist():
    vectorize, rasterize, vectorize_path, rasterize_path = FileManager.import_parameters(FileManager.RND_FRM_CUBST)
    if vectorize == 'True':
        vectorize_cubist_form_to_file(vectorize_path)
    if rasterize == 'True':
        raster_cubist_form_to_file(rasterize_path)
    # convergence, stable, divergence = FileManager.import_parameters(FileManager.RND_FRM_CUBST)
    # if convergence == 'True':
    #     render_form_cubist_phase(FileManager.RND_FRM_CUBST_CONV)
    # if stable == 'True':
    #     render_form_cubist_phase(FileManager.RND_FRM_CUBST_STBL)
    # if divergence == 'True':
    #     render_form_cubist_phase(FileManager.RND_FRM_CUBST_DVRG)
    return


def render_form_smooth_phase(parameters_path):
    vectorize, rasterize, vectorize_path, rasterize_path, os = FileManager.import_parameters(parameters_path)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_contour_from_file(rasterize_path, os)
    return


def render_form_smooth():
    birth, sniffing, wipe, gallop, jump, sit, birth_path, sniffing_path, wipe_path, \
    gallop_path, jump_path, sit_path = FileManager.import_parameters(FileManager.RND_FRM_SMTH)
    if birth == 'True':
        render_form_smooth_phase(birth_path)
    if sniffing == 'True':
        render_form_smooth_phase(sniffing_path)
    if wipe == 'True':
        render_form_smooth_phase(wipe_path)
    if gallop == 'True':
        render_form_smooth_phase(gallop_path)
    if jump == 'True':
        render_form_smooth_phase(jump_path)
    if sit == 'True':
        render_form_smooth_phase(sit_path)
    return


def render_creatures():
    butterfly, bubbles, elevator, arcs, cubes, birds, floor_arcs, square_frame, triangle_frame, rings_pyramid, \
    hollow_rock, yellow_rock, red_rock, first_plank, second_plank, third_plank, tunnel_arcs, wave_rocks, \
    fourth_plank, first_cloud, second_cloud, first_floor, second_floor, third_floor, fifth_plank, big_ring, \
    first_cube_wipe, second_cube_wipe, first_floating_rock, second_floating_rock, third_floating_rock, cliff, \
    butterfly_path, bubbles_path, elevator_path, arcs_path, cubes_path, birds_path, floor_arcs_path, square_frame_path, triangle_frame_path, rings_pyramid_path, \
    hollow_rock_path, yellow_rock_path, red_rock_path, first_plank_path, second_plank_path, third_plank_path, tunnel_arcs_path, wave_rocks_path, \
    fourth_plank_path, first_cloud_path, second_cloud_path, first_floor_path, second_floor_path, third_floor_path, fifth_plank_path, big_ring_path, \
    first_cube_wipe_path, second_cube_wipe_path, first_floating_rock_path, second_floating_rock_path, \
    third_floating_rock_path, cliff_path = FileManager.import_parameters(FileManager.RND_CRTUR)
    if butterfly == 'True':
        render_creature(butterfly_path)
    if bubbles == 'True':
        render_creature(bubbles_path)
    if elevator == 'True':
        render_creature(elevator_path)
    if arcs == 'True':
        render_creature(arcs_path)
    if cubes == 'True':
        render_creature(cubes_path)
    if birds == 'True':
        render_creature(birds_path)
    if floor_arcs == 'True':
        render_creature(floor_arcs_path)
    if square_frame == 'True':
        render_creature(square_frame_path)
    if triangle_frame == 'True':
        render_creature(triangle_frame_path)
    if rings_pyramid == 'True':
        render_creature(rings_pyramid_path)
    if hollow_rock == 'True':
        render_creature(hollow_rock_path)
    if yellow_rock == 'True':
        render_creature(yellow_rock_path)
    if red_rock == 'True':
        render_creature(red_rock_path)
    if first_plank == 'True':
        render_creature(first_plank_path)
    if second_plank == 'True':
        render_creature(second_cloud_path)
    if third_plank == 'True':
        render_creature(third_plank_path)
    if tunnel_arcs == 'True':
        render_creature(tunnel_arcs_path)
    if wave_rocks == 'True':
        render_creature(wave_rocks_path)
    if fourth_plank == 'True':
        render_creature(fourth_plank_path)
    if first_cloud == 'True':
        render_creature(first_cloud_path)
    if second_cloud == 'True':
        render_creature(second_cloud_path)
    if first_floor == 'True':
        render_creature(first_floor_path)
    if second_floor == 'True':
        render_creature(second_floor_path)
    if third_floor == 'True':
        render_creature(third_floor_path)
    if fifth_plank == 'True':
        render_creature(fifth_plank_path)
    if big_ring == 'True':
        render_creature(big_ring_path)
    if first_cube_wipe == 'True':
        render_creature(first_cube_wipe_path)
    if second_cube_wipe == 'True':
        render_creature(second_cube_wipe_path)
    if first_floating_rock == 'True':
        render_creature(first_floating_rock_path)
    if second_floating_rock == 'True':
        render_creature(second_floating_rock_path)
    if third_floating_rock == 'True':
        render_creature(third_floating_rock_path)
    if cliff == 'True':
        render_creature(cliff_path)

