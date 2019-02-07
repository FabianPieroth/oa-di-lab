import numpy as np
from pathlib import Path
import os
import data.data_processing as dp
import logger.visualization as vis
import json
import importlib.util
import pickle


def extract_and_process_logged_folder(folder_name, logging_for_documentation=False):
    print('Read and create input, target and predicted images.')
    rescale_images = True
    json_dict = open_json_file(folder_name=folder_name, file_name='config.json')
    folder_saved_predictions = [s for s in dp.ret_all_files_in_folder(folder_name,
                                                                      full_names=False) if 'predictions' in s]
    data_loader = load_data_loader_module(path=folder_name, json_dict=json_dict)
    if json_dict['do_scale_center']:
        scale_low, scale_high, mean_low, mean_high = load_saved_params(path=folder_name + '/', data_loader=data_loader)
    if json_dict['oa_do_scale_center_before_pca'] and json_dict['oa_do_pca'] and json_dict['image_type'] == 'OA':
        scale_low_before_pca, scale_high_before_pca, mean_low_before_pca, mean_high_before_pca = load_saved_params(path=folder_name + '/',
                                                                                                                   data_loader=data_loader,
                                                                                                                   before_pca=True)

    for folder in folder_saved_predictions:
        data_folder = dp.ret_all_files_in_folder(folder_name + '/' + folder, full_names=False)
        for data in data_folder:
            save_folder = folder_name + '/' 'plots' + '/' + folder + '/' + data
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            image_list = dp.ret_all_files_in_folder(folder_name + '/' + folder + '/' + data, full_names=False)
            for images in image_list:
                input_im, target_im, predict_im = vis.load_file_to_numpy(folder_name + '/' + folder + '/' + data
                                                                         + '/' + images)
                if os.path.exists(folder_name + '/' + folder + '/' + data + '/overlay'):
                    input_us = vis.load_processed_file(full_file_name=folder_name + '/' + folder + '/' + data
                                                       + '/overlay/' + 'US' + images[2:], image_sign='_low')
                    target_us = vis.load_processed_file(full_file_name=folder_name + '/' + folder + '/' + data
                                                        + '/overlay/' + 'US' + images[2:], image_sign='_high')
                else:
                    input_us = None
                    target_us = None
                if json_dict['data_type'] == 'homo':
                    if rescale_images and json_dict['do_scale_center']:
                        input_im, target_im, predict_im = reverse_scaling(input_im=input_im, target_im=target_im,
                                                                          predict_im=predict_im,
                                                                          scale_low=scale_low, scale_high=scale_high,
                                                                          mean_low=mean_low,
                                                                          mean_high=mean_high, data_loader=data_loader)
                    if json_dict['oa_do_pca'] and json_dict['image_type'] == 'OA':
                        input_im, target_im, predict_im = inverse_pca(input_im, target_im, predict_im, path=folder_name,
                                                                      data_loader=data_loader, json_dict=json_dict)

                        if json_dict['oa_do_scale_center_before_pca']:
                            input_im, target_im, predict_im = reverse_scaling(input_im=input_im, target_im=target_im,
                                                                              predict_im=predict_im,
                                                                              scale_low=scale_low_before_pca, scale_high=scale_high_before_pca,
                                                                              mean_low=mean_low_before_pca,
                                                                              mean_high=mean_high_before_pca, data_loader=data_loader)

                vis.plot_channel(input_im, target_im, predict_im, name=images, channel=0,
                                 save_name=save_folder + '/' + images)
                if json_dict['image_type'] == 'OA':
                    if json_dict['oa_do_pca']:
                        json_dict['processing']['channel_slice_oa'] = list(range(28))
                    vis.plot_spectral_test(input_im=input_im, target_im=target_im, predict_im=predict_im, name=images,
                                           input_us=input_us, target_us=target_us, predict_us=target_us,
                                           save_name=save_folder + '/' + images + '_spectra', p_threshold=0.05,
                                           json_processing=json_dict,
                                           logging_for_documentation=logging_for_documentation)
                    vis.plot_single_spectra(input_im=input_im, target_im=target_im, predict_im=predict_im,
                                            input_us=input_us, target_us=target_us, predict_us=target_us,
                                            save_name=save_folder + '/' + images,
                                            slice=json_dict['processing']['channel_slice_oa'],
                                            regressed=json_dict['processing']['use_regressed_oa'],
                                            logging_for_documentation=logging_for_documentation)


def open_json_file(folder_name, file_name):
    with open(folder_name + '/' + file_name, 'r') as file:
        json_dict = json.load(file)
    return json_dict


def reverse_scaling(input_im, target_im, predict_im, scale_low, scale_high, mean_low, mean_high, data_loader):
    input_im = data_loader.scale_and_center_reverse(batch=input_im, scale_params=scale_low,
                                                    mean_image=mean_low)
    target_im = data_loader.scale_and_center_reverse(batch=target_im, scale_params=scale_high,
                                                     mean_image=mean_high)
    predict_im = data_loader.scale_and_center_reverse(batch=predict_im, scale_params=scale_high,
                                                      mean_image=mean_high)
    return input_im, target_im, predict_im


def inverse_pca(input_im, target_im, predict_im, path, data_loader, json_dict):
    pca = data_loader.load_pca_model(path=path)

    input_im = backproject_image_pca(input_im, pca, json_dict)
    target_im = backproject_image_pca(target_im, pca, json_dict)
    predict_im = backproject_image_pca(predict_im, pca, json_dict)

    return input_im, target_im, predict_im


def load_data_loader_module(path, json_dict):
    spec = importlib.util.spec_from_file_location("ProcessData", path + '/data/data_loader.py')
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    # especially trunc_points and do_scale_center
    data_loader = foo.ProcessData(train_ratio=json_dict['train_ratio'], image_type=json_dict['image_type'],
                                  data_type=json_dict['data_type'],
                                  logger_call=True, trunc_points=json_dict['trunc_points'],
                                  do_scale_center=json_dict['do_scale_center'])
    return data_loader


def load_saved_params(path, data_loader, before_pca=False):
    if before_pca:
        param_suffix = '_before_pca'
        trunc_points = data_loader.trunc_points_before_pca
    else:
        param_suffix = ''
        trunc_points = data_loader.trunc_points

    scale_params_low, scale_params_high = data_loader.load_params(param_type="scale_params" + param_suffix,
                                                                  dir_params=path,
                                                                  trunc_points=trunc_points,)

    mean_image_low, mean_image_high = data_loader.load_params(param_type="mean_images" + param_suffix, dir_params=path,
                                                              trunc_points=trunc_points)

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


###########################################
# PCA specific definitions
###########################################

def load_pca_model(folder_name):
    file_path = folder_name + '/OA_pca_model.sav'
    with open(file_path, 'rb') as handle:
        pca_model = pickle.load(handle)
    return pca_model


def project_image_pca(image, pca_model):
    # takes image (C,H,W) and projects it onto the pca space
    # outputs in image shape (pca_comp, H,W)
    # not tested. backproject_image_pca() is tested.
    image = np.moveaxis(image, [0], [-1])
    im_shape = image.shape
    new_shape = list(im_shape[:2])
    new_shape.append(pca_model.n_components)
    image = image.reshape(-1, im_shape[-1])
    transformed = pca_model.transform(image)
    transformed = transformed.reshape(new_shape)
    transformed = np.moveaxis(transformed, [-1], [0])
    return transformed


def backproject_image_pca(pca_image, pca_model, json_dict):
    # takes image in pca components (pca_comp, H,W) and backprojects it into the higher dimensional space
    # outputs in image shape (C,H,W)
    pca_image = np.moveaxis(pca_image, [0], [-1])
    n_comp, n_feat = pca_model.components_.shape
    new_shape = list(pca_image.shape[:2])
    new_shape.append(n_feat)
    pca_image = pca_image.reshape(-1, n_comp)
    if json_dict['pca_use_regress']:
        backproj = np.matmul(pca_image, pca_model.components_)
    else:
        backproj = pca_model.inverse_transform(pca_image)
    backproj = backproj.reshape(new_shape)
    backproj = np.moveaxis(backproj, [-1], [0])
    return backproj


def main():
    path_to_project = str(Path().resolve().parents[1]) + '/reports/'

    logging_for_documentation = True

    folder_name = 'homo/Documentation/combined_model_hyper_1_2019_01_26_16_21_regressed'

    extract_and_process_logged_folder(folder_name=path_to_project + folder_name,
                                      logging_for_documentation=logging_for_documentation)

    plot_train_val_loss(folder_name=path_to_project + folder_name)


if __name__ == "__main__":
    main()
