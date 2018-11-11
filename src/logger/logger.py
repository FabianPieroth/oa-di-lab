import numpy as np
import torch
import os
import datetime
import pickle
import shutil

class Logger(object):
    # This class...
    def __init__(self,
                 project_root_dir,
                 model=None):
        self.project_root_dir = project_root_dir
        self.model = model  # give the model to the logger, so we have all needed variables in the self

        self.base_dir = '%s/reports' % (project_root_dir)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.save_dir = self.base_dir + '/' + self.model.model_name + '_' +  timestamp
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
        with open(self.save_dir + '/' + self.model.model_name + '_mean_images', 'wb') as handle:
            pickle.dump(mean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.save_dir + '/' + self.model.model_name + '_scale_params', 'wb') as handle:
            pickle.dump(scale_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_representation_of_model(self):
        text_file = open(self.save_dir + '/' + self.model.model_name + '_model_structure.txt', "w")
        text_file.write(str(self.model))
        text_file.close()

    def log(self,
            save_appendix,
            current_epoch,
            epochs,
            mean_image_low,
            mean_image_high):
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
            self.save_scale_center_params(mean_images=mean_image_low, scale_params=mean_image_high)
            # copy the data_loader file and the model file to make reproducible examples
            data_loader_path = self.project_root_dir + '/src/data' + '/data_loader.py'
            model_file_path = self.model.model_file_name

            for f in [data_loader_path, model_file_path]:
                shutil.copy(f, self.save_dir)