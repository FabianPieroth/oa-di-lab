import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from logger.oa_spectra_analysis.oa_for_DILab import spectral_F_test, linear_unmixing, _get_default_spectra
import math


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
    agg_size = int(math.ceil(len(train_loss) / nr_epochs))
    fig, ax = plt.subplots()
    ax.plot(np.mean(train_loss.reshape(-1, agg_size), axis=1), label='training loss')
    ax.plot(val_loss, label='validation loss')
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(learning_rates, label='learning rate', c='grey')
    ax2.legend(loc=5)
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')


def plot_spectral_test(input_im, target_im, predict_im, name, save_name, p_threshold, json_processing=None):
    rgb_input = get_relevant_spectra(input_im, p_threshold=p_threshold, json_processing=json_processing)
    rgb_target = get_relevant_spectra(target_im, p_threshold=p_threshold, json_processing=json_processing)
    rgb_predict = get_relevant_spectra(predict_im, p_threshold=p_threshold, json_processing=json_processing)

    if min(rgb_input.shape) == 1:
        rgb_input = rgb_input[:, :, 0]
    if min(rgb_target.shape) == 1:
        rgb_target = rgb_target[:, :, 0]
    if min(rgb_predict.shape) == 1:
        rgb_predict = rgb_predict[:, :, 0]

    plot_and_save_rgb_images(rgb_input, rgb_target, rgb_predict, name, save_name)


def plot_and_save_rgb_images(rgb_input, rgb_target, rgb_predict, name, save_name):
    plt.figure(figsize=(18, 18))
    plt.subplot(1, 3, 1)
    plt.title('input' + '_' + name)
    plt.imshow(rgb_input)

    plt.subplot(1, 3, 2)
    plt.title('target' + '_' + name)
    plt.imshow(rgb_target)

    plt.subplot(1, 3, 3)
    plt.title('predict' + '_' + name)
    plt.imshow(rgb_predict)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')


def get_relevant_spectra(image, p_threshold, json_processing=None):
    if 'use_regressed_oa' not in list(json_processing['processing']):
        print('It should be downward compatible with this, write me if you get errors.')
        p_values = spectral_F_test(np.moveaxis(image, 0, -1))
        reg_coefficients = linear_unmixing(np.moveaxis(image, 0, -1))

        significant_pixels = filter_sign_pixels(p_values=p_values, image=reg_coefficients, p_threshold=p_threshold)

        significant_channel = np.argmax(significant_pixels, axis=2)

        significant_channel[significant_pixels[:, :, 0] == 0] = -1
        significant_channel = significant_channel + 1
        rgb = create_rgb_image(significant_channel)
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
                rgb = create_rgb_image(significant_channel)
                # print('Print the spectrum, the significance mask and the merged images.')
                return rgb
            significant_pixels = image
            significant_channel = np.argmax(significant_pixels, axis=2)
            significant_channel = significant_channel + 1
            rgb = create_rgb_image(significant_channel)
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

            rgb = create_rgb_image(significant_channel)

            return rgb


def filter_sign_pixels(p_values, image, p_threshold):
    p_values[p_values >= p_threshold] = 0.0
    sign_img_pixels = np.zeros(image.shape)
    for i in range(min(image.shape)):
        hb = p_values * image[:, :, i]
        sign_img_pixels[:, :, i] = hb
    return sign_img_pixels


def create_rgb_image(image_2d):
    # set the color values of the spectra, this is not generic, so at the moment only valid for 4 channels!
    palette = np.array([[0, 0, 0],  # not significant: black
                        [148, 0, 211],  # HB: dark violet
                        [220, 20, 60],  # HBO2: crimson
                        [255, 255, 0],  # fat: yellow
                        [30, 144, 255]])  # water: blue
    rgb = palette[image_2d]
    return rgb


def load_file_to_numpy(full_file_name):
    # helper function to load and read the data; pretty inefficient right now
    #  as we need to open every dict two times
    with open(full_file_name, 'rb') as handle:
        sample = pickle.load(handle)
    input_im = np.array([value for key, value in sample.items() if 'input_image' in key][0])
    target_im = np.array([value for key, value in sample.items() if 'target_image' in key][0])
    predict_im = np.array([value for key, value in sample.items() if 'predict_image' in key][0])
    return input_im, target_im, predict_im
