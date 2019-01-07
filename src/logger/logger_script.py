import numpy as np
from pathlib import Path
import os
import data.data_processing as dp
import logger.visualization as vis
import json
import importlib.util


def extract_and_process_logged_folder(folder_name):
    print('Read and create input, target and predicted images.')
    rescale_images = True
    json_dict = open_json_file(folder_name=folder_name, file_name='config.json')
    folder_saved_predictions = [s for s in dp.ret_all_files_in_folder(folder_name,
                                                                      full_names=False) if 'predictions' in s]
    data_loader = load_data_loader_module(path=folder_name, json_dict=json_dict)
    if json_dict['do_scale_center']:
        scale_low, scale_high, mean_low, mean_high = load_saved_params(path=folder_name + '/', data_loader=data_loader)

    for folder in folder_saved_predictions:
        data_folder = dp.ret_all_files_in_folder(folder_name + '/' + folder, full_names=False)
        for data in data_folder:
            save_folder = folder_name + '/' 'plots' + '/' + folder + '/' + data
            os.makedirs(save_folder)
            image_list = dp.ret_all_files_in_folder(folder_name + '/' + folder + '/' + data, full_names=False)
            for images in image_list:
                input_im, target_im, predict_im = vis.load_file_to_numpy(folder_name + '/' + folder + '/' + data
                                                                         + '/' + images)
                if rescale_images and json_dict['do_scale_center']:
                    """input_im = data_loader.scale_and_center_reverse(batch=input_im, scale_params=scale_low,
                                                                    mean_image=mean_low)
                    target_im = data_loader.scale_and_center_reverse(batch=target_im, scale_params=scale_high,
                                                                     mean_image=mean_high)
                    predict_im = data_loader.scale_and_center_reverse(batch=predict_im, scale_params=scale_high,
                                                                      mean_image=mean_high)"""
                    pass
                vis.plot_channel(input_im, target_im, predict_im, name=images, channel=0,
                                 save_name=save_folder + '/' + images)
                if json_dict['image_type'] == 'OA':
                    vis.plot_spectral_test(input_im=input_im, target_im=target_im, predict_im=predict_im, name=images,
                                           save_name=save_folder + '/' + images + '_spectra', p_threshold=0.05,
                                           json_processing=json_dict)


def open_json_file(folder_name, file_name):
    with open(folder_name + '/' + file_name, 'r') as file:
        json_dict = json.load(file)
    return json_dict


def load_data_loader_module(path, json_dict):
    spec = importlib.util.spec_from_file_location("ProcessData", path + '/data/data_loader.py')
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    # TODO: Add additional parameters from the json dict in newer versions
    # especially trunc_points and do_scale_center
    data_loader = foo.ProcessData(train_ratio=json_dict['train_ratio'], image_type=json_dict['image_type'],
                                  data_type=json_dict['data_type'],
                                  logger_call=True, trunc_points=json_dict['trunc_points'],
                                  do_scale_center=json_dict['do_scale_center'])
    return data_loader


def load_saved_params(path, data_loader):

    scale_params_low, scale_params_high = data_loader.load_params(param_type="scale_params", dir_params=path,
                                                                  trunc_points=data_loader.trunc_points)

    mean_image_low, mean_image_high = data_loader.load_params(param_type="mean_images", dir_params=path,
                                                              trunc_points=data_loader.trunc_points)

    return scale_params_low, scale_params_high, mean_image_low, mean_image_high


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

    folder_name = 'hetero/combined_model_hyper_1_2019_01_07_17_08'

    extract_and_process_logged_folder(folder_name=path_to_project + folder_name)

    plot_train_val_loss(folder_name=path_to_project + folder_name)


if __name__ == "__main__":
    main()
