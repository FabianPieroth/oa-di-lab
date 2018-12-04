import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_channel(im_input, im_target, im_predict, name, channel):
    plt.figure(figsize=(18, 18))
    plt.subplot(1, 3, 1)
    plt.title('input' + '_' + name)
    plt.imshow(im_input[channel, :, :], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('target' + '_' + name)
    plt.imshow(im_target[channel, :, :], cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('predict' + '_' + name)
    plt.imshow(im_predict[channel, :, :], cmap='gray')


def plot_oa_spectra(im_input, im_target, im_predict, x, y, name, figsize=(18, 5)):
    for i in range(2):
        print(im_predict.shape)
        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.title('input ' + name + '_pixel:' + str(x + i) + ', ' + str(y))
        plt.plot(im_input[:,x + i, y,])

        plt.subplot(1, 3, 2)
        plt.title('target ' + name + '_pixel:' + str(x + i) + ', ' + str(y))
        plt.plot(im_target[:, x + i, y])

        plt.subplot(1, 3, 3)
        plt.title('predicted ' + name + '_pixel:' + str(x + i) + ', ' + str(y))
        plt.plot(im_predict[:, x + i, y])


def load_file_to_numpy(full_file_name):
    # helper function to load and read the data; pretty inefficient right now
    #  as we need to open every dict two times
    with open(full_file_name, 'rb') as handle:
        sample = pickle.load(handle)
    input_im = np.array([value for key, value in sample.items() if 'input_image' in key][0])
    target_im = np.array([value for key, value in sample.items() if 'target_image' in key][0])
    predict_im = np.array([value for key, value in sample.items() if 'predict_image' in key][0])
    return input_im, target_im, predict_im
