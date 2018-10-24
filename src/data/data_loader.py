import sys
import numpy as np
sys.path.append("..")


class PreprocessImages(object):
    """
    TODO: Description of data loader in general; keep it growing
    train_ratio:    Gives the train and validation split for the model
    """

    def __init__(self, train_ratio):
        # initialize and write into self, then call the prepare data and return the data to the trainer
        self.train_ratio = train_ratio

        # self.X_train, self.y_train, self.X_test, self.y_test, self.df_complete = self._prepare_data()

    def _prepare_data(self):
        # load data
        pass
        # return X_train, y_train, X_test, y_test, df_complete

    def _split_data(self, df):
        train_size = int(len(df) * self.train_ratio)
        df_train, df_test = df[:train_size], df[train_size:]
        return df_train, df_test

    def _partition_for_cnn(self, df):
        # returns the data in the format for the model
        pass
        # return ...

    def _load_processed_data(self):
        # load the already preprocessed data
        pass
        # return df
