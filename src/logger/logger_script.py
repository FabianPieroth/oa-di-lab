import numpy as np
from pathlib import Path
import os
import data.data_processing as dp
import logger.visualization as vis
import json


def extract_and_process_logged_folder(folder_name):
    print('Read and create input, target and predicted images.')
    json_dict = open_json_file(folder_name=folder_name, file_name='config.json')
    folder_saved_predictions = [s for s in dp.ret_all_files_in_folder(folder_name,
                                                                      full_names=False) if 'predictions' in s]
    for folder in folder_saved_predictions:
        data_folder = dp.ret_all_files_in_folder(folder_name + '/' + folder, full_names=False)
        for data in data_folder:
            save_folder = folder_name + '/' 'plots' + '/' + folder + '/' + data
            os.makedirs(save_folder)
            image_list = dp.ret_all_files_in_folder(folder_name + '/' + folder + '/' + data, full_names=False)
            for images in image_list:
                input_im, target_im, predict_im = vis.load_file_to_numpy(folder_name + '/' + folder + '/' + data
                                                                         + '/' + images)
                vis.plot_channel(input_im, target_im, predict_im, name=images, channel=0,
                                 save_name=save_folder + '/' + images)
                if json_dict['image_type'] == 'OA':
                    vis.plot_spectral_test(input_im=input_im, target_im=target_im, predict_im=predict_im, name=images,
                                           save_name=save_folder + '/' + images + '_spectra', p_threshold=0.01,
                                           json_processing=json_dict)


def open_json_file(folder_name, file_name):
    with open(folder_name + '/' + file_name, 'r') as file:
        json_dict = json.load(file)
    return json_dict


def plot_train_val_loss(folder_name):
    print('Read and plot train-validation loss.')
    json_dict = open_json_file(folder_name=folder_name, file_name='config.json')
    model_name = json_dict['model_name']
    nr_epochs = json_dict['nr_epochs']
    train_loss = np.load(folder_name + '/' + model_name + '_train_loss.npy')
    val_loss = np.load(folder_name + '/' + model_name + '_validation_loss.npy')
    learning_rates = json_dict['learning_rates']

    vis.plot_train_val_loss_graph(train_loss=train_loss, val_loss=val_loss, learning_rates=learning_rates,
                                  save_name=folder_name + '/plots/' + 'train_val_loss', nr_epochs=nr_epochs)


def main():
    path_to_project = str(Path().resolve().parents[1]) + '/reports/'
    folder_name = 'homo/2018_12_14_OA_all_only_regressed'
    extract_and_process_logged_folder(folder_name=path_to_project + folder_name)

    plot_train_val_loss(folder_name=path_to_project + folder_name)


if __name__ == "__main__":
    main()
