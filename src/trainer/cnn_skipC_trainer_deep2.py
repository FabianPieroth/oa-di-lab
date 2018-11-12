from data.data_loader import ProcessData
import trainer.utils as utils
from logger.logger import Logger
import numpy as np
from models import awesomeImageTranslatorJunior
import torch
import torch.nn as nn
import datetime

class CNN_skipCo_trainer(object):
    def __init__(self):

        self.dataset = ProcessData(train_ratio=0.9,process_raw_data=False,
                                   do_augment=False, add_augment=True,
                                   do_flip=True, do_blur=True, do_deform=True, do_crop=True,
                                   image_type='US', get_scale_center=True, single_sample=False)

        self.model = awesomeImageTranslatorJunior.AwesomeImageTranslatorJunior(
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam,
            learning_rate=0.001,
            weight_decay=0
        )

        if torch.cuda.is_available():
            torch.cuda.current_device()
            self.model.cuda()

        self.logger = Logger()

    def fit(self, epochs=10):
        # get scale and center parameters
        scale_params_low, scale_params_high = utils.load_params(image_type=self.dataset.image_type,
                                                                param_type="scale_params")
        mean_image_low, mean_image_high = utils.load_params(image_type=self.dataset.image_type,
                                                            param_type="mean_images")

        # currently for one image:
        '''
        self.dataset.batch_names(batch_size=5)
        X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[1])
        print(X.shape)
        X = X[0,:,:]
        Y = Y[0,:,:]
        '''
        X_val, Y_val = self.dataset.create_train_batches(self.dataset.val_file_names)

        scale_center_X_val = utils.scale_and_center(X_val, scale_params_low, mean_image_low,
                                                    image_type=self.dataset.image_type)
        scale_center_Y_val = utils.scale_and_center(Y_val, scale_params_high, mean_image_high,
                                                    image_type=self.dataset.image_type)

        scale_center_X_val = np.array([scale_center_X_val])
        scale_center_Y_val = np.array([scale_center_Y_val])

        # (C, N, H, W) to (N, C, H, W)
        scale_center_X_val = scale_center_X_val.reshape(scale_center_X_val.shape[1], scale_center_X_val.shape[0],
                                                        scale_center_X_val.shape[2], scale_center_X_val.shape[3])
        scale_center_Y_val = scale_center_Y_val.reshape(scale_center_Y_val.shape[1], scale_center_Y_val.shape[0],
                                                        scale_center_Y_val.shape[2], scale_center_Y_val.shape[3])

        input_tensor_val, target_tensor_val = torch.from_numpy(scale_center_X_val), torch.from_numpy(scale_center_Y_val)

        for e in range(0, epochs):
            # separate names into random batches and shuffle every epoch
            self.dataset.batch_names(batch_size=32)
            # in self.batch_number is the number of batches in the training set
            for i in range(self.dataset.batch_number):
                X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[i])

                # scale and center the batch
                scale_center_X = utils.scale_and_center(X, scale_params_low, mean_image_low,
                                                        image_type=self.dataset.image_type)
                scale_center_Y = utils.scale_and_center(Y, scale_params_high, mean_image_high,
                                                        image_type=self.dataset.image_type)

                scale_center_X = np.array([scale_center_X])
                scale_center_Y = np.array([scale_center_Y])

                # (C, N, H, W) to (N, C, H, W)
                scale_center_X = scale_center_X.reshape(scale_center_X.shape[1], scale_center_X.shape[0],
                                                        scale_center_X.shape[2], scale_center_X.shape[3])
                scale_center_Y = scale_center_Y.reshape(scale_center_Y.shape[1], scale_center_Y.shape[0],
                                                        scale_center_Y.shape[2], scale_center_Y.shape[3])


                input_tensor, target_tensor = torch.from_numpy(scale_center_X), torch.from_numpy(scale_center_Y)

                if torch.cuda.is_available():

                    input_tensor = input_tensor.cuda()
                    target_tensor = target_tensor.cuda()
                    input_tensor_val = input_tensor_val.cuda()
                    target_tensor_val = target_tensor_val.cuda()

                self.model.train_model(input_tensor, target_tensor, current_epoch=e)

                # save model every x epochs
            if e % 100 == 0:
                self.logger.save_model(self.model, model_name=self.model.model_name +'_' + str(datetime.datetime.now())+'_epoch_' + str(e))
                self.logger.save_loss(self.model.model_name, self.model.train_loss, self.model.val_loss)

            if e == 0:
                self.logger.save_scale_center_params(model_name=self.model.model_name,
                                                     mean_images=[mean_image_low, mean_image_high],
                                                     scale_params=[scale_params_low, scale_params_high])
                self.logger.save_representation_of_model(self.model.model_name, str(self.model))

            # write validation loss per epoch
            self.model.get_val_loss(val_in=input_tensor_val, val_target=target_tensor_val)

                ## how to undo the scaling:
                #unscaled_X = utils.scale_and_center_reverse(scale_center_X, scale_params_low, mean_image_low, image_type = self.dataset.image_type)
                #unscaled_Y = utils.scale_and_center_reverse(scale_center_Y, scale_params_high, mean_image_high, image_type=self.dataset.image_type)

    def predict(self):
        #self.model.predict()

        # see self.dataset.X_val and self.dataset.Y_val
        pass

    def log_model(self, model_name=None):
        self.logger.log(self.model,
                        model_name=model_name,
                        train_loss=self.model.train_loss,
                        model_structure=str(self.model))


def main():
    trainer = CNN_skipCo_trainer()

    #fit the first model
    print('---------------------------')
    print('fitting deep model')
    trainer.fit(epochs=10)
    trainer.predict()
    #torch.save(trainer.model, "../../reports/model.pt")
    trainer.log_model(model_name=trainer.model.model_name)
    print('\n---------------------------')


if __name__ == "__main__":
    main()
