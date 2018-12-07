from data.data_loader import ProcessData
import trainer.utils as utils
from logger.logger_module import Logger
import numpy as np
from models import cnn_skipC_model, awesomeImageTranslator1000
import torch
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt

class CNN_skipCo_trainer(object):
    def __init__(self):

        self.dataset = ProcessData(train_ratio=0.3,process_raw_data=False,
                                   do_augment=False, image_type='US', get_scale_center=False, single_sample=True)

        self.model = cnn_skipC_model.cnn_skipC_model(
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam,
            learning_rate=0.01,
            weight_decay=0
        )

        self.model_2 = awesomeImageTranslator1000.AwesomeImageTranslator1000(
            criterion=nn.MSELoss(),
            optimizer=torch.optim.Adam,
            learning_rate=0.01,
            weight_decay=0
        )

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
        for e in range(0, epochs):
            # separate names into random batches and shuffle every epoch
            self.dataset.batch_names(batch_size=5)
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
                    #print('CUDA available')
                    #print('current device ' + str(cur_dev))
                    #print('device count ' + str(torch.cuda.device_count()))
                    #print('device name ' + torch.cuda.get_device_name(cur_dev))

                    cur_dev = torch.cuda.current_device()
                    input_tensor.cuda()
                    target_tensor.cuda()


                #self.model.train_model(input_tensor, target_tensor, current_epoch=e)
                self.model.train_model(input_tensor, target_tensor, current_epoch=e)

                # save model every x epochs
                if e % 5 == 0:
                    self.logger.save_model(self.model, model_name=self.model.model_name +'_' + str(datetime.datetime.now())+'_epoch_' + str(e))
                    self.logger.save_loss(self.model.model_name, self.model.train_loss, self.model.test_loss)


                    # save model every 50 epochs
                    #if e % 50 == 0:
                    #    self.logger.save_model(self.model, model_name=self.model.model_name + '_epoch_' + str(e))



                if e == 0:
                    self.logger.save_scale_center_params(model_name=self.model.model_name,
                                                             mean_images=[mean_image_low, mean_image_high],
                                                             scale_params=[scale_params_low, scale_params_high])
                    self.logger.save_representation_of_model(self.model.model_name, str(self.model))

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
    print('fitting first model')
    trainer.fit(epochs=50)
    trainer.predict()
    #torch.save(trainer.model, "../../reports/model.pt")
    trainer.log_model(model_name=trainer.model.model_name)
    print('\n---------------------------')

    #fit the second model
    print('---------------------------')
    print('fitting second model')
    trainer.model = trainer.model_2
    print(trainer.model.model_name)
    trainer.fit(epochs=15)
    trainer.log_model(model_name=trainer.model.model_name)
    print('\n---------------------------')
    print('---------------------------')


if __name__ == "__main__":
    main()
