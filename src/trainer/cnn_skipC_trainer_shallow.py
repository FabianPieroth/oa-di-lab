from data.data_loader import ProcessData
from logger.logger import Logger
import numpy as np
from models import cnn_skipC_model
import torch
import torch.nn as nn


class CNN_skipCo_trainer(object):
    def __init__(self):

        self.dataset = ProcessData(train_ratio=0.9, process_raw_data=False,
                                   do_augment=False, add_augment=True,
                                   do_flip=True, do_blur=True, do_deform=True, do_crop=True,
                                   image_type='US', get_scale_center=False, single_sample=True)

        self.model = cnn_skipC_model.cnn_skipC_model(
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam,
            learning_rate=0.001,
            weight_decay=0
        )

        if torch.cuda.is_available():
            torch.cuda.current_device()
            self.model.cuda()

        self.logger = Logger(model=self.model, project_root_dir=self.dataset.project_root_dir)
        self.epochs = 50

    def fit(self):
        # get scale and center parameters
        scale_params_low, scale_params_high = self.dataset.load_params(param_type="scale_params")
        mean_image_low, mean_image_high = self.dataset.load_params(param_type="mean_images")

        # currently for one image:
        '''
        self.dataset.batch_names(batch_size=5)
        X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[1])
        print(X.shape)
        X = X[0,:,:]
        Y = Y[0,:,:]
        '''
        # load validation set, normalize and parse into tensor

        input_tensor_val, target_tensor_val = self.dataset.scale_and_parse_to_tensor(
                                                    batch_files=self.dataset.val_file_names,
                                                    scale_params_low=scale_params_low,
                                                    scale_params_high=scale_params_high,
                                                    mean_image_low=mean_image_low,
                                                    mean_image_high=mean_image_high)

        if torch.cuda.is_available():
            input_tensor_val = input_tensor_val.cuda()
            target_tensor_val = target_tensor_val.cuda()

        for e in range(0, self.epochs):
            # separate names into random batches and shuffle every epoch
            self.dataset.batch_names(batch_size=32)
            # in self.batch_number is the number of batches in the training set
            for i in range(self.dataset.batch_number):

                input_tensor, target_tensor = self.dataset.scale_and_parse_to_tensor(
                                                batch_files=self.dataset.val_file_names,
                                                scale_params_low=scale_params_low,
                                                scale_params_high=scale_params_high,
                                                mean_image_low=mean_image_low,
                                                mean_image_high=mean_image_high)

                if torch.cuda.is_available():

                    input_tensor = input_tensor.cuda()
                    target_tensor = target_tensor.cuda()

                self.model.train_model(input_tensor, target_tensor, current_epoch=e)

            # calculate the validation loss and add to validation history
            self.logger.get_val_loss(val_in=input_tensor_val, val_target=target_tensor_val)
            # save model every x epochs
            if e % 25 == 0 or e == self.epochs - 1:
                self.logger.log(save_appendix='_epoch_' + str(e),
                                current_epoch=e,
                                epochs=self.epochs,
                                mean_image_low=mean_image_low,
                                mean_image_high=mean_image_high)

                # how to undo the scaling:
                # unscaled_X = utils.scale_and_center_reverse(scale_center_X,
                #  scale_params_low, mean_image_low, image_type = self.dataset.image_type)
                # unscaled_Y = utils.scale_and_center_reverse(scale_center_Y, scale_params_high,
                #  mean_image_high, image_type=self.dataset.image_type)

    def predict(self):
        # self.model.predict()

        # see self.dataset.X_val and self.dataset.Y_val
        pass


def main():
    trainer = CNN_skipCo_trainer()

    # fit the first model
    print('---------------------------')
    print('fitting shallow model')
    trainer.fit()
    trainer.predict()
    # torch.save(trainer.model, "../../reports/model.pt")
    # trainer.log_model(model_name=trainer.model.model_name)
    print('\n---------------------------')


if __name__ == "__main__":
    main()
