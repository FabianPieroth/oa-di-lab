import torch
import torch.nn as nn
import numpy as np
import json

#import matplotlib

#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt


def load_torch_model(path_to_load, name_to_load, data_loader, model, starting_point, pass_complete_input):
    print('Loading saved model from starting point ' + str(starting_point) + ' and model ' + str(name_to_load) + '.')
    project_root_dir = data_loader.project_root_dir
    # overwrite epochs, learning rate, train and val files
    json_dict = open_json_file(folder_name=project_root_dir + '/' + path_to_load, file_name='config.json')
    loaded_learning_rates = json_dict['learning_rates']
    loaded_train_files = json_dict['train_files']
    loaded_val_names = json_dict['val_files']

    if not pass_complete_input:
        data_loader.train_file_names = loaded_train_files
        data_loader.val_file_names = loaded_val_names
        loaded_learning_rates = np.roll(loaded_learning_rates, -starting_point)
        # load model and initialize the weights
        # model = nn.DataParallel(model)
        # state_parallel = model.state_dict()
        pretrained_dict = torch.load(project_root_dir + '/' + path_to_load + '/' + name_to_load #, map_location='cpu'
         )
        model.load_state_dict(pretrained_dict)
    else:
        loaded_learning_rates = np.roll(loaded_learning_rates, -starting_point)
        # load model and initialize the weights
        # model = nn.DataParallel(model)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(project_root_dir + '/' + path_to_load + '/' + name_to_load, # map_location='cpu'
                                  )
        # name faulty module: module.models.0.last_layers.0.deconv.weight
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                            k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # load a test file to check if model was correctly loaded

    '''scale_params_low, scale_params_high = data_loader.load_params(param_type="scale_params")
    mean_image_low, mean_image_high = data_loader.load_params(param_type="mean_images")
    input_tensor, target_tensor = data_loader.scale_and_parse_to_tensor(
        batch_files=data_loader.test_names[0:2],
        scale_params_low=scale_params_low,
        scale_params_high=scale_params_high,
        mean_image_low=mean_image_low,
        mean_image_high=mean_image_high)
    model.eval()
    predict = model(input_tensor)
    predict_new = predict.detach().cpu().numpy()[0, :, :, :]

    plt.imshow(predict_new[0, :, :], cmap='gray')'''

    return data_loader, model, loaded_learning_rates


def open_json_file(folder_name, file_name):
    with open(folder_name + '/' + file_name, 'r') as file:
        json_dict = json.load(file)
    return json_dict