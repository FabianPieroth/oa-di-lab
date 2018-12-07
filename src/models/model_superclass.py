import sys
import torch
import torch.nn as nn


class ImageTranslator(nn.Module):

    def __init__(self, models, criterion=nn.MSELoss(), optimizer=torch.optim.Adam,
                 model_name='combined_model', ):
        """
        This class trains the models given. It can be used to stack the models in modular fashion
        :param models: list of image translating models to be concatenated in that order
        """
        super().__init__()
        self.models = nn.ModuleList([model for model in models])

        self.model_file_name = __file__  # save file name to copy file in logger into logging folder
        self.model_name = model_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loss = []
        self.val_loss = []

    def activate_optimizer(self, learning_rate):
        self.optimizer = self.optimizer(self.parameters(), lr=learning_rate)

    def forward(self, X):
        for model in self.models:
            X = model.forward(X)
        return X

    def train_model(self, X, y, current_epoch=None):
        """
        Trains the model for one batch
        :param X: batch input
        :param y: batch target
        :param current_epoch: the current epoch for displaying in print statement
        """
        def closure():
            self.optimizer.zero_grad()
            out = self.forward(X)
            loss = self.criterion(out, y)
            if current_epoch is not None:
                sys.stdout.write('\r' + ' epoch ' + str(current_epoch) +
                                 ' |  loss : ' + str(loss.item()))
            else:
                sys.stdout.write('\r  loss : ' + str(loss.item()))
            self.train_loss.append(loss.item())
            loss.backward()
            return loss
        self.optimizer.step(closure)

    def set_learning_rate(self, learning_rate):
        """
        setting the learning rate. Can be explicitly done to have another learning rate without new initialization
        :param learning_rate: learning rate to be set
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate