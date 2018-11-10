import json
import numpy as np
import torch
from pathlib import Path

class Logger(object):
    # This class...
    def __init__(self):
        # CHANGE FOR TRAINING HERE ROOT DIRECTORY
        # project_root_dir = '/mnt/local/mounted'
        project_root_dir = Path().resolve().parents[1]

        self.base_dir = '%s/reports' % (project_root_dir)

    def save_model(self, model, model_name):
        # This Method should save the model in a serialized folder structure
        # Serialize a model and its weights into json and h5 file.
        # serialize model to JSON
        torch.save(model, self.base_dir+'/model_'+ model_name +'.pt')

    def load_model(self, model_name):
        # Load a saved model
        # Serialize a model and its weights into json and h5 file.
        # load json and create model
        print("Loaded model from disk")
        return torch.load(self.base_dir+'/model_' + model_name + '.pt')

    def calculate_metrics(self, model):
        # Methods calculates all the metrics that we want for the evaluation
        # so more than just the metric that we optimize during training
        pass

    def save_series_in_csv(self, model):
        # saves the predicted images on the hard drive to evaluate later on, needs a bool
        pass

    def save_loss(self, train_loss, valid_loss=None, model_name=None):
        if model_name is None:
            model_name = ""
        np.save(self.base_dir + '/'+model_name+'train_loss', train_loss)

        if valid_loss is not None:
            np.save(self.base_dir + '/'+model_name+'validation_loss', valid_loss)

    def save_representation_of_model(self, data, model_name=None):
        if model_name is None:
            model_name = 'model'
        text_file = open(self.base_dir+'/'+model_name+'_structure.txt', "w")
        text_file.write(data)
        text_file.close()

    def log(self,
            model=None,
            model_name=None,
            train_loss=None,
            test_loss=None,
            model_structure=None):
        # method to call the other methods and decide what should be saved, this should be called in the trainer

        if model is not None:
            if model_name is None:
                model_name = 'final'

            self.save_model(model, model_name)

        if train_loss is not None:
            self.save_loss(train_loss, test_loss, model_name)

        if model_structure is not None:
            self.save_representation_of_model(model_structure, model_name)
