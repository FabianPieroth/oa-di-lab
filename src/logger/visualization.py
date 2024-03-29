import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pickle
from logger.oa_spectra_analysis.oa_for_DILab import spectral_F_test, linear_unmixing, _get_default_spectra
import math
import sys
from logger.oa_spectra_analysis.generate_unmixing_figures import generate_figure as gf
import os


def plot_channel(im_input, im_target, im_predict, name, channel=None, save_name=None):
    if channel is None:
        in_channel = min(im_input.shape) - 1
        tar_channel = min(im_target.shape) - 1
        pre_channel = min(im_predict.shape) - 1
    else:
        in_channel = channel
        tar_channel = channel
        pre_channel = channel
    plt.figure(figsize=(18, 18))
    plt.subplot(1, 3, 1)
    plt.title('input' + '_' + name)
    plt.imshow(im_input[in_channel, :, :], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('target' + '_' + name)
    plt.imshow(im_target[tar_channel, :, :], cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('predict' + '_' + name)
    plt.imshow(im_predict[pre_channel, :, :], cmap='gray')

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
        plt.clf()
        plt.close('all')


'''def plot_oa_spectra(im_input, im_target, im_predict, x, y, name, figsize=(18, 5)):
    for i in range(2):
        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.title('input ' + name + '_pixel:' + str(x + i) + ', ' + str(y))
        plt.plot(im_input[:,x + i, y,])

        plt.subplot(1, 3, 2)
        plt.title('target ' + name + '_pixel:' + str(x + i) + ', ' + str(y))
        plt.plot(im_target[:, x + i, y])

        plt.subplot(1, 3, 3)
        plt.title('predicted ' + name + '_pixel:' + str(x + i) + ', ' + str(y))
        plt.plot(im_predict[:, x + i, y])'''


def plot_train_val_loss_graph(train_loss, val_loss, learning_rates, nr_epochs, save_name=None):
    if len(val_loss)+1 < nr_epochs:
        aggregate_by = len(val_loss)
    else:
        aggregate_by = nr_epochs
    agg_size = int(math.ceil(len(train_loss) / aggregate_by))
    fig, ax = plt.subplots()
    ax.plot(np.mean(train_loss.reshape(-1, agg_size), axis=1), label='Training loss')
    ax.plot(val_loss, label='Validation loss')
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    ax2 = ax.twinx()
    ax2.plot(learning_rates, label='Learning Rate', c='grey')
    ax2.set_ylabel("Learning Rate")
    ax2.legend(loc=5)
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
        plt.clf()
        plt.close('all')


def plot_spectral_test(input_im, target_im, predict_im, name, save_name, p_threshold, json_processing=None,
                       logging_for_documentation=False, input_us=None, target_us=None, predict_us=None):
    rgb_input = get_relevant_spectra(input_im, p_threshold=p_threshold, json_processing=json_processing)
    rgb_target = get_relevant_spectra(target_im, p_threshold=p_threshold, json_processing=json_processing)
    rgb_predict = get_relevant_spectra(predict_im, p_threshold=p_threshold, json_processing=json_processing)

    if min(rgb_input.shape) == 1:
        rgb_input = rgb_input[:, :, 0]
    if min(rgb_target.shape) == 1:
        rgb_target = rgb_target[:, :, 0]
    if min(rgb_predict.shape) == 1:
        rgb_predict = rgb_predict[:, :, 0]

    plot_and_save_rgb_images(rgb_input, rgb_target, rgb_predict, name, save_name,
                             logging_for_documentation=logging_for_documentation,
                             input_us=input_us, target_us=target_us, predict_us=target_us)


def plot_and_save_rgb_images(rgb_input, rgb_target, rgb_predict, name, save_name, logging_for_documentation,
                             input_us=None, target_us=None, predict_us=None):
    input_us = prepare_us_for_overlay(us=input_us, rgb=rgb_input)
    target_us = prepare_us_for_overlay(us=target_us, rgb=rgb_target)
    predict_us = prepare_us_for_overlay(us=predict_us, rgb=rgb_predict)

    rgb_input = overlay_us_rgb(us=input_us, rgb=rgb_input)
    rgb_input = overlay_us_rgb(us=target_us, rgb=rgb_target)
    rgb_input = overlay_us_rgb(us=predict_us, rgb=rgb_predict)


    if logging_for_documentation:
        f, axarr = plt.subplots(1, 3, figsize=(18, 18))
        axarr[0].imshow(input_us, extent=(-.5, rgb_input.shape[1] - .5, rgb_input.shape[0] - .5, -.5), cmap='gray')
        axarr[0].imshow(rgb_input, norm=PowerNorm(gamma=.5))
        axarr[0].axis('off')
        axarr[0].set_title('Input')

        axarr[1].imshow(target_us, extent=(-.5, rgb_target.shape[1] - .5, rgb_target.shape[0] - .5, -.5), cmap='gray')
        axarr[1].imshow(rgb_target, norm=PowerNorm(gamma=.5))
        axarr[1].axis('off')
        axarr[1].set_title('Target')

        axarr[2].imshow(predict_us, extent=(-.5, rgb_predict.shape[1] - .5, rgb_predict.shape[0] - .5, -.5), cmap='gray')
        axarr[2].imshow(rgb_predict, norm=PowerNorm(gamma=.5))
        axarr[2].axis('off')
        axarr[2].set_title('Predict')
    else:
        plt.figure(figsize=(18, 18))
        plt.subplot(1, 3, 1)
        plt.title('input' + '_' + name)
        plt.imshow(rgb_input, norm=PowerNorm(gamma=.5))

        plt.subplot(1, 3, 2)
        plt.title('target' + '_' + name)
        plt.imshow(rgb_target, norm=PowerNorm(gamma=.5))

        plt.subplot(1, 3, 3)
        plt.title('predict' + '_' + name)
        plt.imshow(rgb_predict, norm=PowerNorm(gamma=.5))

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
        plt.clf()
        plt.close('all')


def prepare_us_for_overlay(us, rgb):
    if us is None:
        us = np.zeros((rgb.shape[0], rgb.shape[1]))
    else:
        if np.max(us) == np.min(us):
            us = us - np.min(us)
        else:
            us = (us - np.min(us)) / (np.max(us) - np.min(us))
    return us[:rgb.shape[0], :rgb.shape[1]]


def overlay_us_rgb(us, rgb):
    for i in range(rgb.shape[2]):
        rgb[:, :, i] = np.maximum(us, rgb[:, :, i])
    return rgb


def get_relevant_spectra(image, p_threshold, json_processing=None):
    if 'processing' not in list(json_processing):
        print('It should be downward compatible with this, write me if you get errors.')
        p_values = spectral_F_test(np.moveaxis(image, 0, -1))
        reg_coefficients = linear_unmixing(np.moveaxis(image, 0, -1))

        significant_pixels = filter_sign_pixels(p_values=p_values, image=reg_coefficients, p_threshold=p_threshold)

        significant_channel = np.argmax(significant_pixels, axis=2)

        significant_channel[significant_pixels[:, :, 0] == 0] = -1
        significant_channel = significant_channel + 1
        rgb = create_rgb_image(image_2d=significant_channel, image_3d=significant_pixels)
        return rgb
    else:
        if json_processing['processing']['use_regressed_oa']:
            image = np.moveaxis(image, 0, -1)
            if json_processing['processing']['only_f_test_in_target']:
                # print('Only print the significance mask of input, target and predicted.')
                if min(image.shape) > 1:
                    image = np.expand_dims(image[:, :, 0], axis=2)
                image[image >= p_threshold] = 1.0
                image[image < p_threshold] = 0.0
                return image
            if json_processing['processing']['add_f_test']:
                p_values = image[:, :, 0]
                reg_coefficients = image[:, :, 1:]
                significant_pixels = filter_sign_pixels(p_values=p_values, image=reg_coefficients,
                                                        p_threshold=p_threshold)
                significant_channel = np.argmax(significant_pixels, axis=2)
                significant_channel[significant_pixels[:, :, 0] == 0] = -1
                significant_channel = significant_channel + 1
                rgb = create_rgb_image(image_2d=significant_channel, image_3d=significant_pixels)
                # print('Print the spectrum, the significance mask and the merged images.')
                return rgb
            significant_pixels = image
            significant_channel = np.argmax(significant_pixels, axis=2)
            significant_channel = significant_channel + 1
            rgb = create_rgb_image(image_2d=significant_channel, image_3d=significant_pixels)
            return rgb

        else:
            channel_oa_slice = json_processing['processing']['channel_slice_oa']
            spectra = _get_default_spectra()
            spectra_slice = spectra[:, channel_oa_slice]
            # image_sliced = image[channel_oa_slice, :, :]

            p_values = spectral_F_test(np.moveaxis(image, 0, -1), spectra=spectra_slice)
            reg_coefficients = linear_unmixing(np.moveaxis(image, 0, -1), spectra=spectra_slice)

            significant_pixels = filter_sign_pixels(p_values=p_values, image=reg_coefficients, p_threshold=p_threshold)

            significant_channel = np.argmax(significant_pixels, axis=2)

            significant_channel[significant_pixels[:, :, 0] == 0] = -1
            significant_channel = significant_channel + 1

            rgb = create_rgb_image(image_2d=significant_channel, image_3d=significant_pixels)

            return rgb


def filter_sign_pixels(p_values, image, p_threshold):
    sign_mask = np.zeros(p_values.shape)
    sign_mask[p_values >= p_threshold] = 0.0
    sign_mask[p_values < p_threshold] = 1.0
    sign_img_pixels = np.zeros(image.shape)
    for i in range(min(image.shape)):
        hb = sign_mask * image[:, :, i]
        sign_img_pixels[:, :, i] = hb
    return sign_img_pixels


def create_rgb_image(image_2d, image_3d=None, coloring_type='continuous'):
    # coloring_type:
    #       'continuous': add the blood channels and display it as rgb image
    #       'discrete': color the maximum value of the array
    #       'blood': only display the blood spectra
    # set the color values of the spectra, this is not generic, so at the moment only valid for 4 channels!
    if coloring_type == 'discrete':
        palette = np.array([[0, 0, 0],  # not significant: black
                            [148, 0, 211],  # HB: dark violet
                            [220, 20, 60],  # HBO2: crimson
                            [255, 255, 0],  # fat: yellow
                            [30, 144, 255]])  # water: blue
        rgb = palette[image_2d]
    elif coloring_type == 'continuous':
        if np.min(image_3d.shape) > 4:
            sys.exit('There is no continuous spectra for the input available, check the parameters.')
        # increase the brightness in depth
        image_3d = exp_increasing_with_depth(image_3d)
        image_3d = np.maximum(0, image_3d)
        rgb = np.zeros((image_3d.shape[0], image_3d.shape[1], 3))
        rgb[:, :, 0] = image_3d[:, :, 0] + image_3d[:, :, 1]
        rgb[:, :, [1,2]] = image_3d[:, :, [2,3]]
        rgb[:, :, 0] = rgb[:, :, 0] / np.max(rgb[:, :, 0])
        rgb[:, :, 1] = rgb[:, :, 1] / np.max(rgb[:, :, 1])
        rgb[:, :, 2] = rgb[:, :, 2] / np.max(rgb[:, :, 2])
        # rgb = rgb - np.min(rgb) + 1.0  # shift to positive scale, so that we can use the log transform
        # rgb = np.log(rgb)  # log scale
        # rgb = (rgb - np.min(rgb))/ (np.max(rgb) - np.min(rgb))  # normalize for displaying the image to [0,1] range
        # rgb = rgb / np.max(rgb)
    else:
        print('to be implemented')
    return rgb


def exp_increasing_with_depth(image, constant=0.0):
    # this is so inefficient it hurts, but it was fast to implement
    factors = np.arange(start=0, stop=image.shape[0])
    factors = np.exp(constant * factors)
    factors = np.repeat(np.expand_dims(factors, axis=1), repeats=image.shape[1], axis=1)
    factors = np.repeat(np.expand_dims(factors, axis=2), repeats=image.shape[2], axis=2)

    return image * factors


def plot_single_spectra(input_im, target_im, predict_im, save_name, regressed,
                        input_us=None, target_us=None, predict_us=None, slice=None, logging_for_documentation=False):
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    input_us = prepare_us_for_overlay(us=input_us, rgb=input_im[0, :, :])
    target_us = prepare_us_for_overlay(us=target_us, rgb=target_im[0, :, :])
    predict_us = prepare_us_for_overlay(us=predict_us, rgb=predict_im[0, :, :])

    fig1 = gf(img=input_im, us=input_us, slice=slice, regressed=regressed,
              logging_for_documentation=logging_for_documentation)
    fig2 = gf(img=target_im, us=target_us, slice=slice, regressed=regressed,
              logging_for_documentation=logging_for_documentation)
    fig3 = gf(img=predict_im, us=predict_us, slice=slice, regressed=regressed,
              logging_for_documentation=logging_for_documentation)

    # save the plots
    fig1.savefig(save_name + '/' + 'Input', bbox_inches='tight')
    fig2.savefig(save_name + '/' + 'Target', bbox_inches='tight')
    fig3.savefig(save_name + '/' + 'Predict', bbox_inches='tight')
    plt.clf()
    plt.close('all')


def load_file_to_numpy(full_file_name):
    # helper function to load and read the data; pretty inefficient right now
    #  as we need to open every dict two times
    with open(full_file_name, 'rb') as handle:
        sample = pickle.load(handle)
    input_im = np.array([value for key, value in sample.items() if 'input_image' in key][0])
    target_im = np.array([value for key, value in sample.items() if 'target_image' in key][0])
    predict_im = np.array([value for key, value in sample.items() if 'predict_image' in key][0])
    return input_im, target_im, predict_im


def load_processed_file(full_file_name, image_sign):
    # helper function to load and read the data; pretty inefficient right now
    #  as we need to open every dict two times
    with open(full_file_name, 'rb') as handle:
        sample = pickle.load(handle)
    sample_array = [value for key, value in sample.items() if image_sign in key][0]
    return sample_array
