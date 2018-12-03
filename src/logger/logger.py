import numpy as np
import torch
import os
import datetime
import pickle
import shutil
import json

class Logger(object):
    # This class...
    def __init__(self,
                 project_root_dir,
                 dataset,
                 epochs,
                 batch_size,
                 learning_rates,
                 model=None,
                 image_type='US'):
        self.project_root_dir = project_root_dir
        self.model = model  # give the model to the logger, so we have all needed variables in the self
        self.image_type = image_type
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rates = learning_rates

        self.base_dir = '%s/reports' % (project_root_dir)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

        self.save_dir = self.base_dir + '/' + self.dataset.data_type + '/' + self.model.model_name + '_' +  timestamp
        self.load_dir = self.base_dir + '/' + self.dataset.data_type + '/' + self.model.model_name
        os.makedirs(self.save_dir)

    def save_model(self, save_appendix):
        # This Method should save the model in a serialized folder structure
        # Serialize a model and its weights into json and h5 file.
        # serialize model to JSON
        torch.save(self.model.state_dict(), self.save_dir + '/' + self.model.model_name + '_' + save_appendix + '.pt')

    '''
    This Method has to be redone, after we have a certain saving structure!
    
    def load_model(self, save_appendix):
        # Load a saved model
        # load json and create model
        # this only works if the file with the class is not altered after the save!
        print("Loaded model from disk")
        return torch.load(self.save_dir + '/' + self.model.model_name + '_' + save_appendix + '.pt')
    '''

    def load_predict(self, save_appendix, time_stamp, input_tensor, target_tensor):
        self.model.load_state_dict(torch.load(self.load_dir + '_' + time_stamp + '/' + self.model.model_name + '__' + save_appendix + '.pt'))
        # set the state of the model to eval()
        self.model.eval()
        predict = self.model(input_tensor) # self.model.predict(input_tensor)
        return predict


    def save_us_channel(self, im_input, im_target, im_predict, i, time_stamp):

        input = {"input_image"+str(i) : im_input}
        target = {"target_image"+str(i) : im_target}
        predict = {"predict_image"+str(i) : im_predict}
        with open(self.load_dir + '_' + time_stamp + '/input', 'wb') as handle:
            pickle.dump(input, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.load_dir + '_' + time_stamp + '/target', 'wb') as handle:
            pickle.dump(target, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.load_dir + '_' + time_stamp + '/predict', 'wb') as handle:
            pickle.dump(predict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
            scale_params):
        # method to call the other methods and decide what should be saved, this should be called in the trainer

        if self.model is not None:
            if current_epoch == epochs - 1:
                save_appendix = 'final'
            self.save_model(save_appendix=save_appendix)

        if self.model.train_loss is not None:
            self.save_loss()

        if str(self.model) is not None:
            self.save_representation_of_model()

        if current_epoch == 0:
            # only call this the first time, when training starts
            self.save_representation_of_model()
            self.save_scale_center_params(mean_images=mean_images, scale_params=scale_params)
            self.save_json_file(batch_size=self.batch_size, epochs=self.epochs,
                                       learning_rates=self.learning_rates)

            # copy the data_loader file and the model file to make reproducible examples
            data_loader_path = self.project_root_dir + '/src/data/' + '/data_loader.py'
            model_file_path = self.model.model_file_name
            augmentation_file_path = self.project_root_dir + '/src/data' + '/augmentation.py'
            data_processing_file_path = self.project_root_dir + '/src/data' +'/data_processing.py'

            os.makedirs(self.save_dir + '/data')
            os.makedirs(self.save_dir + '/models')

            # copy the files in the corresponding folder
            shutil.copy(data_loader_path, self.save_dir + '/data')
            shutil.copy(augmentation_file_path, self.save_dir + '/data')
            shutil.copy(data_processing_file_path, self.save_dir + '/data')
            shutil.copy(model_file_path, self.save_dir + '/models')

    def save_json_file(self, batch_size, epochs,learning_rates):

        config = {


            "batch_size" : self.batch_size,
            "model_name" : self.model.model_name,
            "image_type" : self.dataset.image_type,
            "data_type" : self.dataset.data_type,
            "nr_epochs" : self.epochs,
            "applied_augmentations" : {
                "process_raw_data" : self.dataset.process_raw,
                "do_augment" : self.dataset.do_augment,
                "add_augment" : self.dataset.add_augment ,
                "do_flip" : self.dataset.do_flip,
                "do_blur" : self.dataset.do_blur,
                "do_deform" : self.dataset.do_deform,
                "do_crop" : self.dataset.do_crop,
                "image_type" : self.dataset.image_type,
                "data_type" : self.dataset.data_type,
                "get_scale_center" : self.dataset.get_scale_center,
                "single_sample" : self.dataset.single_sample
            },
            "train_valid_split" : self.dataset.train_ratio,
            "loss_function" : str(self.model.criterion),
            "train_files" : self.dataset.train_file_names,
            "val_files" : self.dataset.val_file_names,
            "learning_rates": self.learning_rates
        }
        file_path = self.save_dir + '/config.json'
        with open(file_path, 'w') as outfile:
            json.dump(config, outfile)





