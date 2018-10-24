import json
import numpy as np
from matplotlib.path import Path


class Logger(object):
    # This class...
    def __init__(self, save_model = False):

        self.save_model = save_model
        PROJECT_ROOT_DIR = Path().resolve().parents[1]
        self.base_dir = '%s/reports/results' % (PROJECT_ROOT_DIR)

    def save_model(self, model):
        # This Method should save the model in a serialized folder structure
        # Serialize a model and its weights into json and h5 file.
        # serialize model to JSON
        pass

    def load_model(self):
        # Load a saved model
        # Serialize a model and its weights into json and h5 file.
        # load json and create model
        pass
        print("Loaded model from disk")
        #return loaded_model

    def calculate_metrics(self, model):
        # Methods calculates all the metrics that we want for the evaluation
        # so more than just the metric that we optimize during training
        pass

    def save_series_in_csv(self, model):
        # saves the predicted images on the hard drive to evaluate later on, needs a bool
        pass

    def log(self):
        # method to call the other methods and decide what should be saved, this should be called in the trainer
        pass

