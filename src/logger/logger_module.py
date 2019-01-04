import numpy as np
import torch
import os
import datetime
import pickle
import shutil
import json
import random


class Logger(object):
    def __init__(self,
                 project_root_dir,
                 dataset,
                 epochs,
                 batch_size,
                 learning_rates,
                 hyper_no,
                 model=None,
                 image_type='US'):
        self.project_root_dir = project_root_dir
        self.model = model  # give the model to the logger, so we have all needed variables in the self
        self.image_type = image_type
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rates = learning_rates

        self.save_appendix = None

        self.base_dir = '%s/reports' % project_root_dir
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

        self.save_dir = self.base_dir + '/' + self.dataset.data_type + '/' + self.model.model_name + '_'+ 'hyper_' + str(hyper_no+1) + '_' + timestamp
        self.load_dir = self.base_dir + '/' + self.dataset.data_type + '/' + self.model.model_name
        os.makedirs(self.save_dir)

    def save_model(self):
        # This Method should save the model in a serialized folder structure
        # Serialize a model and its weights into json and h5 file.
        # serialize model to JSON
        torch.save(self.model.state_dict(), self.save_dir + '/' + self.model.model_name
                   + 'model' + self.save_appendix + '.pt')

    def predict_eval_images(self, mean_images, scale_params, num_images_train=2, num_images_val=3):

        train_length = min(len(self.dataset.train_file_names), num_images_train)
        train_names = random.sample(self.dataset.train_file_names, train_length)

        val_length = min(len(self.dataset.val_file_names), num_images_val)
        val_names = random.sample(self.dataset.val_file_names, val_length)

        test_names = self.dataset.test_names

        # load scale and center params
        scale_params_low = scale_params[0]
        scale_params_high = scale_params[1]
        mean_image_low = mean_images[0]
        mean_image_high = mean_images[1]

        # save validation image predictions
        self.load_and_forward(names=val_names, scale_params_low=scale_params_low,
                              scale_params_high=scale_params_high, mean_image_low=mean_image_low,
                              mean_image_high=mean_image_high, image_class='val')

        # save train image predictions
        self.load_and_forward(names=train_names, scale_params_low=scale_params_low,
                              scale_params_high=scale_params_high, mean_image_low=mean_image_low,
                              mean_image_high=mean_image_high, image_class='train')

        # save test image predictions
        if not len(test_names) == 0:
            self.load_and_forward(names=test_names, scale_params_low=scale_params_low,
                                  scale_params_high=scale_params_high, mean_image_low=mean_image_low,
                                  mean_image_high=mean_image_high, image_class='test')
        else:
            print('There are no files in the processed test_set folder, so no files are logged here.')

    def load_and_forward(self, names, scale_params_low, scale_params_high, mean_image_low,
                         mean_image_high, image_class):

        input_tensor, target_tensor = self.dataset.scale_and_parse_to_tensor(
            batch_files=names,
            scale_params_low=scale_params_low,
            scale_params_high=scale_params_high,
            mean_image_low=mean_image_low,
            mean_image_high=mean_image_high)

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # forward with model and get prediction tensor
        predict = self.predict(input_tensor)

        for i in range(input_tensor.shape[0]):
            input_save = input_tensor.detach().cpu().numpy()[i, :, :, :]
            target_save = target_tensor.detach().cpu().numpy()[i, :, :, :]
            predict_new = predict.detach().cpu().numpy()[i, :, :, :]

            '''if self.dataset.do_scale_center:
                input_save = self.dataset.scale_and_center_reverse(input_save, scale_params_low,
                                                                    mean_image_low)
                target_save = self.dataset.scale_and_center_reverse(target_save, scale_params_high,
                                                                    mean_image_high)
                predict_new = self.dataset.scale_and_center_reverse(predict_new, scale_params_high,
                                                                    mean_image_high)'''

            self.save_predictions(input_save=input_save, target_save=target_save, predict_save=predict_new,
                                  save_name=self.dataset.extract_name_from_path(names[i], without_ch=False),
                                  image_class=image_class)

    def predict(self, x):
        self.model.eval()
        predict_image = self.model(x)
        self.model.train()

        return predict_image

    def save_predictions(self, input_save, target_save, predict_save, save_name, image_class):
        dict_save = {"input_image": input_save,
                     "target_image": target_save,
                     "predict_image": predict_save}
        save_dir = self.save_dir + '/predictions' + self.save_appendix + '/' + image_class
        os.makedirs(save_dir, exist_ok=True)

        with open(save_dir + '/' + save_name, 'wb') as handle:
            pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    def load_predict(self, save_appendix, time_stamp, input_tensor, target_tensor):
        self.model.load_state_dict(torch.load(self.load_dir + '_' + time_stamp + '/' +
         self.model.model_name + '__' + save_appendix + '.pt'))
        # set the state of the model to eval()
        self.model.eval()
        predict = self.model(input_tensor) # self.model.predict(input_tensor)
        return predict
    '''

    def calculate_metrics(self, model):
        # Methods calculates all the metrics that we want for the evaluation
        # so more than just the metric that we optimize during training
        pass

    def get_val_loss(self, val_in=None, val_target=None):
        # do a forward pass with validation set for every epoch and get validation loss
        if val_in is not None and val_target is not None:
            with torch.no_grad():
                val_out = self.model.forward(val_in)
                val_loss = self.model.criterion(val_out, val_target)
                self.model.val_loss.append(val_loss.item())
        return

    def save_loss(self):
        np.save(self.save_dir + '/' + self.model.model_name + '_train_loss', np.array(self.model.train_loss))

        if self.model.val_loss is not None:
            np.save(self.save_dir + '/' + self.model.model_name + '_validation_loss', np.array(self.model.val_loss))

    def save_scale_center_params(self, mean_images, scale_params):
        # write files into dictionary
        scale_param = {self.image_type + '_low': scale_params[0], self.image_type + '_high': scale_params[1]}
        mean_image = {self.image_type + '_low': mean_images[0], self.image_type + '_high': mean_images[1]}
        with open(self.save_dir + '/' + self.image_type + '_mean_images', 'wb') as handle:
            pickle.dump(mean_image, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.save_dir + '/' + self.image_type + '_scale_params', 'wb') as handle:
            pickle.dump(scale_param, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_representation_of_model(self):
        text_file = open(self.save_dir + '/' + self.model.model_name + '_model_structure.txt', "w")
        text_file.write(str(self.model))
        text_file.close()

    def log(self,
            save_appendix,
            current_epoch,
            epochs,
            mean_images,
            scale_params,
            learning_rates):
        # method to call the other methods and decide what should be saved, this should be called in the trainer
        self.save_appendix = save_appendix

        if self.model is not None:
            if current_epoch == epochs - 1:
                self.save_appendix = 'final'
            self.save_model()

        if self.model.train_loss is not None:
            self.save_loss()

        self.predict_eval_images(mean_images=mean_images, scale_params=scale_params)

        # save the learning rates into self

        self.learning_rates = learning_rates

        if current_epoch == 0:
            # only call this the first time, when training starts
            self.save_representation_of_model()
            self.save_scale_center_params(mean_images=mean_images, scale_params=scale_params)
            self.save_json_file()

            # copy the data_loader file and the model file to make reproducible examples
            data_loader_path = self.project_root_dir + '/src/data/' + '/data_loader.py'
            model_file_path = self.model.model_file_name
            augmentation_file_path = self.project_root_dir + '/src/data' + '/augmentation.py'
            data_processing_file_path = self.project_root_dir + '/src/data' + '/data_processing.py'
            visualizer_path = self.project_root_dir + '/src/' + '/logger/visualization.py'

            os.makedirs(self.save_dir + '/data')
            os.makedirs(self.save_dir + '/models')

            # copy the files in the corresponding folder
            shutil.copy(data_loader_path, self.save_dir + '/data')
            shutil.copy(augmentation_file_path, self.save_dir + '/data')
            shutil.copy(data_processing_file_path, self.save_dir + '/data')
            shutil.copy(visualizer_path, self.save_dir + '/data')
            shutil.copy(model_file_path, self.save_dir + '/models')

    def save_json_file(self):

        config = {
            "batch_size": self.batch_size,
            "model_name": self.model.model_name,
            "image_type": self.dataset.image_type,
            "train_ratio": self.dataset.train_ratio,
            "data_type": self.dataset.data_type,
            "nr_epochs": self.epochs,
            "do_scale_center": self.dataset.do_scale_center,
            "get_scale_center": self.dataset.get_scale_center,
            "trunc_points": self.dataset.trunc_points,
            "applied_augmentations": {
                "process_raw_data": self.dataset.process_raw,
                "do_augment": self.dataset.do_augment,
                "add_augment": self.dataset.add_augment,
                "do_flip": self.dataset.do_flip,
                "do_blur": self.dataset.do_blur,
                "do_deform": self.dataset.do_deform,
                "do_crop": self.dataset.do_crop,
                'do_speckle_noise': self.dataset.do_speckle_noise,
                "image_type": self.dataset.image_type,
                "data_type": self.dataset.data_type,
                "hetero_mask_to_mask": self.dataset.hetero_mask_to_mask,
                "single_sample": self.dataset.single_sample
            },
            #'model_parameters': {
            #    'conv_channels': self.model.conv_channels,
            #    'strides': self.model.strides,
            #    'kernels': self.model.kernels,
            #    'padding': self.model.padding,
            #    'output_padding': self.model.output_padding
                # 'criterion': self.model.criterion,  is not JSON serializable
                # 'optimizer': self.model.optimizer  is not JSON serializable
            #},
            'processing': {
                'height_channel_oa': self.dataset.height_channel_oa,
                'use_regressed_oa': self.dataset.use_regressed_oa,
                'include_regression_error': self.dataset.include_regression_error,
                'add_f_test': self.dataset.add_f_test,
                'only_f_test_in_target': self.dataset.only_f_test_in_target,
                'channel_slice_oa': self.dataset.channel_slice_oa
            },
            "train_valid_split": self.dataset.train_ratio,
            "loss_function": str(self.model.criterion),
            "train_files": self.dataset.train_file_names,
            "val_files": self.dataset.val_file_names,
            "learning_rates": self.learning_rates.tolist()
        }
        file_path = self.save_dir + '/config.json'
        with open(file_path, 'w') as outfile:
            json.dump(config, outfile)
