import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from logger.oa_spectra_analysis.oa_for_DILab import spectral_F_test


def plot_channel(im_input, im_target, im_predict, name, channel, save_name=None):

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

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')


def plot_oa_spectra(im_input, im_target, im_predict, x, y, name, figsize=(18, 5)):
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
        plt.plot(im_predict[:, x + i, y])


def plot_train_val_loss_graph(train_loss, val_loss, learning_rates, nr_epochs, save_name=None):
    agg_size = int(len(train_loss) / nr_epochs)
    fig, ax = plt.subplots()
    ax.plot(np.mean(train_loss.reshape(-1, agg_size), axis=1), label='training loss')
    ax.plot(val_loss, label='validation loss')
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(learning_rates, label='learning rate', c='grey')
    ax2.legend(loc=5)
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')


def plot_spectral_test(input_im, target_im, predict_im, name, save_name):
    pvalue_input = spectral_F_test(input_im)
    print(pvalue_input)


def load_file_to_numpy(full_file_name):
    # helper function to load and read the data; pretty inefficient right now
    #  as we need to open every dict two times
    with open(full_file_name, 'rb') as handle:
        sample = pickle.load(handle)
    input_im = np.array([value for key, value in sample.items() if 'input_image' in key][0])
    target_im = np.array([value for key, value in sample.items() if 'target_image' in key][0])
    predict_im = np.array([value for key, value in sample.items() if 'predict_image' in key][0])
    return input_im, target_im, predict_im
