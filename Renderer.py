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
        im_rgb = Colourizer.colour_volume(shape, r / 255, g / 255, b / 255, int(tlr), int(blr_k), clr_s)
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
        im_bcp = Vectorizer.vectorize_image(im_y, min_crv_ratio)
        # Saving the Bezier control points to a file.
        bcp_f_name = FileManager.file_path(dir_out_bcp_path, f_prefix, n_padded, 'txt', os)
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
            stroke = Rasterizer.stroke_rasterizer(cur_bcp, strk_w_min, strk_w_max, texture=cur_txr, canvas_shape=cnvs_shape)
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
        im_id = f'{f_prefix}.{n_padded}'

        # Preparing the image.
        im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
        im = FileManager.import_image(im_name)

        idx_factor = 1
        if scale == 'True':
            idx_factor = Vectorizer.index_displace_distort_factor(im_file_idx, start, end, dsp_dst_direction, dsp_dst_style)
        # Size in the frame.

        # strk_w_max = 0.5 * content_radius
        # Rastering the curves.

        im_rgb, im_a = Rasterizer.content_rasterizer(im, cnvs_shape, canvas_scaler, idx_factor, displace,
                                                     displace_transform_max, strk_w_min, 3, 1, r_min, r_max, g_min,
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


# def raster_content_from_file(parameters_path, os='w'):
#     dir_in_path, f_prefix, dir_out_path, start, end, digits_num, canvas_input_x, canvas_input_y, canvas_output_x,\
#     canvas_output_y, scale, scale_factor, contiguous, diminish, diminish_min_l_r, dsp_dst_direction, dsp_dst_style, \
#     displace, displace_min, displace_max, displace_transform_max, distort, distort_min, distort_max, strk_w_min, \
#     strk_w_max, texture_style, texture_type, colour_style, r_min, r_max, g_min, g_max, b_min, b_max, alpha, alpha_c, \
#     alpha_f = FileManager.import_parameters(parameters_path)
#     cnvs_shape = (int(canvas_output_x), int(canvas_output_y))
#     canvas_scaler = canvas_output_x / canvas_input_x
#     # Iterating over the desired images.
#     for im_file_idx in range(int(start), int(end) + 1):
#         # Preparing the input/output file name.
#         n_padded = f'{im_file_idx}'
#         while len(n_padded) < digits_num:
#             n_padded = f'0{n_padded}'
#         im_name = f'{f_prefix}.{n_padded}'
#         # Preparing the image.
#         im_r = np.zeros(cnvs_shape)
#         im_g = np.zeros(cnvs_shape)
#         im_b = np.zeros(cnvs_shape)
#         im_a = np.zeros(cnvs_shape)
#         # Importing the Bezier control points from the text file.
#         bcp_f_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'txt', os)
#         bcp_arr, bcp_num = FileManager.import_bezier_control_points(bcp_f_name)
#         # Checking if there are any Bezier control points to raster.
#         if bcp_num < 1:
#             im_rgb = np.dstack((im_r, im_g, im_b))
#             FileManager.save_rgba_image(dir_out_path, im_name, im_rgb, im_a, os)
#             continue
#         bcp_arr *= canvas_scaler
#         # Applying vector manipulation.
#         idx_factor = Vectorizer.index_displace_distort_factor(im_file_idx, start, end, dsp_dst_direction, dsp_dst_style)
#         if diminish == 'True':
#             bcp_arr = Rasterizer.diminish_bcps_num(bcp_arr, float(diminish_min_l_r))
#         if displace == 'True':
#             dsp_f = int(displace_min + idx_factor * (displace_max - displace_min))
#             dsp_t = int(idx_factor * displace_transform_max)
#             bcp_arr = Vectorizer.displace_bezier_curves(bcp_arr, dsp_f, dsp_t)
#         if distort == 'True':
#             dst_f = int(distort_min + idx_factor * (distort_max - distort_min))
#             bcp_arr = Vectorizer.distort_bezier_curves(bcp_arr, dst_f)
#         # Size in the frame.
#         center = 0.5 * (np.min(bcp_arr, axis=(1, 0)) + np.max(bcp_arr, axis=(1, 0)))
#         content_radius = np.max(np.linalg.norm(center - bcp_arr, axis=2))
#         # strk_w_max = 0.5 * content_radius
#         # Rastering the curves.
#         curves_num = len(bcp_arr)
#         txr_arr = Rasterizer.generate_textures_arr(curves_num, texture_style, texture_type)  # Defining texture.
#         rgb_range = np.array([[r_min, r_max], [g_min, g_max], [b_min, b_max]], dtype=int)
#         clr_arr = Colourizer.generate_colours_arr(curves_num, colour_style, rgb_range)  # Defining colour.
#         for crv_idx in range(curves_num):
#             cur_bcp = bcp_arr[crv_idx]
#             cur_txr = txr_arr[crv_idx]
#             cur_clr = clr_arr[crv_idx]
#             # strk_w_min = 1
#             norm_to_center = np.min(np.linalg.norm(center - cur_bcp, axis=1))
#             strk_w_max = 0.125 * (content_radius - norm_to_center) + strk_w_min
#             stroke = Rasterizer.stroke_rasterizer(cur_bcp, strk_w_min, strk_w_max, texture=cur_txr,
#                                                   canvas_shape=cnvs_shape)
#             stroke_rgb = np.repeat(stroke, Colourizer.CLR_DIM).reshape((int(canvas_output_x), int(canvas_output_y),
#                                                                         Colourizer.CLR_DIM))
#             stroke_rgb = Colourizer.colour_stroke(stroke_rgb, cur_clr[0], cur_clr[1], cur_clr[2])
#             stroke_alpha_im = Colourizer.alpha_channel(stroke, alpha, alpha_c, int(alpha_f))
#             stroke_im_r = stroke_rgb[::, ::, :1:].reshape(cnvs_shape)
#             stroke_im_g = stroke_rgb[::, ::, 1:2:].reshape(cnvs_shape)
#             stroke_im_b = stroke_rgb[::, ::, 2::].reshape(cnvs_shape)
#             im_r = Colourizer.composite_rgb(stroke_im_r, im_r, stroke_alpha_im)
#             im_g = Colourizer.composite_rgb(stroke_im_g, im_g, stroke_alpha_im)
#             im_b = Colourizer.composite_rgb(stroke_im_b, im_b, stroke_alpha_im)
#             im_a = Colourizer.composite_alpha(stroke_alpha_im, im_a)
#         im_rgb = np.dstack((im_r, im_g, im_b))
#         FileManager.save_rgba_image(dir_out_path, im_name, im_rgb, im_a, os)


# def raster_content_from_file0(parameters_path, os='w'):
#     dir_in_path, f_prefix, dir_out_path, start, end, digits_num, canvas_input_x, canvas_input_y, canvas_output_x,\
#     canvas_output_y, scale, scale_factor, contiguous, diminish, diminish_min_l_r, dsp_dst_direction, dsp_dst_style, \
#     displace, displace_min, displace_max, displace_transform_max, distort, distort_min, distort_max, strk_w_min, \
#     strk_w_max, texture_style, texture_type, colour_style, r_min, r_max, g_min, g_max, b_min, b_max, alpha, alpha_c, \
#     alpha_f = FileManager.import_parameters(parameters_path)
#     cnvs_shape = (int(canvas_output_x), int(canvas_output_y))
#     canvas_scaler = canvas_output_x / canvas_input_x
#     # Iterating over the desired images.
#     for im_file_idx in range(int(start), int(end) + 1):
#         # Preparing the input/output file name.
#         n_padded = f'{im_file_idx}'
#         while len(n_padded) < digits_num:
#             n_padded = f'0{n_padded}'
#         im_name_out = f'{f_prefix}.{n_padded}'
#         im_name = FileManager.file_path(dir_in_path, f_prefix, n_padded, 'png', os)
#         im = FileManager.import_image(im_name)
#         im_yiq = Colourizer.rgb_to_yiq(im)
#         alpha_intput_im = Colourizer.alpha_input_image(im)
#         im_y = im_yiq[:, :, 0]
#         # Finding the Bezier control points.
#         bcp_arr, bcp_num, init_edge_pxl_num = Rasterizer.content_rasterizer(im_y, im_file_idx)
#         # Preparing the image.
#         im_r = np.zeros(cnvs_shape)
#         im_g = np.zeros(cnvs_shape)
#         im_b = np.zeros(cnvs_shape)
#         im_a = np.zeros(cnvs_shape)
#         # Checking if there are any Bezier control points to raster.
#         if bcp_num < 1:
#             im_rgb = np.dstack((im_r, im_g, im_b))
#             FileManager.save_rgba_image(dir_out_path, im_name, im_rgb, im_a, os)
#             continue
#         bcp_arr = (bcp_arr * canvas_scaler)
#         # Applying vector manipulation.
#         idx_factor = Vectorizer.index_displace_distort_factor(im_file_idx, start, end, dsp_dst_direction, dsp_dst_style)
#         if diminish == 'True':
#             bcp_arr = Rasterizer.diminish_bcps_num(bcp_arr, float(diminish_min_l_r))
#         if displace == 'True':
#             dsp_f = int(displace_min + idx_factor * (displace_max - displace_min))
#             dsp_t = int(idx_factor * displace_transform_max)
#             bcp_arr = Vectorizer.displace_bezier_curves(bcp_arr, dsp_f, dsp_t)
#         if distort == 'True':
#             dst_f = int(distort_min + idx_factor * (distort_max - distort_min))
#             bcp_arr = Vectorizer.distort_bezier_curves(bcp_arr, dst_f)
#         # Size in the frame.
#         pxl_num = len(np.argwhere(alpha_intput_im != 0))
#         relative_size = pxl_num / (im_y.shape[0] * im_y.shape[1])
#         # Rastering the curves.
#         curves_num = len(bcp_arr)
#         txr_arr = Rasterizer.generate_textures_arr(curves_num, texture_style, texture_type)  # Defining texture.
#         rgb_range = np.array([[r_min, r_max], [g_min, g_max], [b_min, b_max]], dtype=int)
#         clr_arr = Colourizer.generate_content_colours_arr(curves_num, colour_style, rgb_range)  # Defining colour.
#         content_radius = np.sqrt(pxl_num / np.pi)
#         strk_w_min = 1
#         strk_w_max = content_radius * 0.5
#         # cur_strk_w_max = content_radius * relative_size * canvas_scaler + strk_w_min
#         # if cur_strk_w_max > content_radius:
#         #     strk_w_max
#         for crv_idx in range(curves_num):
#             cur_bcp = bcp_arr[crv_idx]
#             cur_txr = txr_arr[crv_idx]
#             cur_clr = clr_arr[crv_idx]
#             stroke = Rasterizer.stroke_rasterizer(cur_bcp, strk_w_min, strk_w_max, texture=cur_txr, canvas_shape=cnvs_shape)
#             stroke_rgb = np.repeat(stroke, Colourizer.CLR_DIM).reshape((int(canvas_output_x), int(canvas_output_y),
#                                                                         Colourizer.CLR_DIM))
#             stroke_rgb = Colourizer.colour_stroke(stroke_rgb, cur_clr[0], cur_clr[1], cur_clr[2])
#             stroke_alpha_im = Colourizer.alpha_channel(stroke, alpha, alpha_c, int(alpha_f))
#             stroke_im_r = stroke_rgb[::, ::, :1:].reshape(cnvs_shape)
#             stroke_im_g = stroke_rgb[::, ::, 1:2:].reshape(cnvs_shape)
#             stroke_im_b = stroke_rgb[::, ::, 2::].reshape(cnvs_shape)
#             im_r = Colourizer.composite_rgb(stroke_im_r, im_r, stroke_alpha_im)
#             im_g = Colourizer.composite_rgb(stroke_im_g, im_g, stroke_alpha_im)
#             im_b = Colourizer.composite_rgb(stroke_im_b, im_b, stroke_alpha_im)
#             im_a = Colourizer.composite_alpha(stroke_alpha_im, im_a)
#         im_rgb = np.dstack((im_r, im_g, im_b))
#         FileManager.save_rgba_image(dir_out_path, im_name_out, im_rgb, im_a, os)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Text ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def render_text(parameters_path):
    vectorize, rasterize, colourize, vectorize_path, rasterize_path, colourize_path, os = FileManager.import_parameters(
        parameters_path)
    if vectorize == 'True':
        vectorize_contour_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_text_contour_from_file(rasterize_path, os)
    if colourize == 'True':
        volume_colourizer(colourize_path, os)
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


# ---------------------------------------------------- Content ---------------------------------------------------------
def render_content(parameters_path):
    vectorize, rasterize, colourize, vectorize_path, rasterize_path, colourize_path, os = FileManager.import_parameters(
        parameters_path)
    if vectorize == 'True':
        vectorize_content_to_file(vectorize_path, os)
    if rasterize == 'True':
        raster_content_from_file(rasterize_path, os)
    if colourize == 'True':
        volume_colourizer(colourize_path, os)
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


def render_content_setup():
    render_content(FileManager.RND_CNT_SETUP)
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
