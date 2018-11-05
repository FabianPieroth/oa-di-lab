from data.data_loader import ProcessData
import trainer.utils as utils
import numpy as np
from models import cnn_skipC_model
import torch
import torch.nn as nn

class CNN_skipCo_trainer(object):
    def __init__(self):

        self.dataset = ProcessData(train_ratio=0.3,process_raw_data=True, do_augment=False, image_type='US', get_scale_center=True)

        self.model = cnn_skipC_model.cnn_skipC_model(
            criterion=nn.MSELoss(),
            optimizer= torch.optim.Adam,
            learning_rate=0.001,
            weight_decay=0
        )


        #self.logger = Logger(self)

    def fit(self):
        self.dataset.batch_names(batch_size=5) # call this to separate names into random batches
        # get scale and center parameters
        scale_params_low, scale_params_high = utils.load_params(image_type=self.dataset.image_type, param_type="scale_params")
        mean_image_low, mean_image_high = utils.load_params(image_type=self.dataset.image_type, param_type="mean_images")
        print(mean_image_low.shape)

        epochs=100
        for e in range(0, epochs):
            # in self.batch_number is the number of batches in the training set
            for i in range(self.dataset.batch_number):
                X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[i])
                # scale and center the batch
                scale_center_X = utils.scale_and_center(X, scale_params_low, mean_image_low, image_type=self.dataset.image_type)
                scale_center_Y = utils.scale_and_center(Y, scale_params_high, mean_image_high, image_type=self.dataset.image_type)
                scale_center_X = np.array([scale_center_X])
                scale_center_Y = np.array([scale_center_Y])
                #print(scale_center_Y.shape)
                # (C, N, H, W) to (N, C, H, W)
                scale_center_X = scale_center_X.reshape(scale_center_X.shape[1], scale_center_X.shape[0], scale_center_X.shape[2], scale_center_X.shape[3])
                scale_center_Y = scale_center_Y.reshape(scale_center_Y.shape[1], scale_center_Y.shape[0], scale_center_Y.shape[2], scale_center_Y.shape[3])
                #print(scale_center_X.shape)
                #print(scale_center_Y.shape)

                input_tensor, target_tensor = torch.from_numpy(scale_center_X), torch.from_numpy(scale_center_Y)
                #print('------------ Testing CUDA ----------')
                if torch.cuda.is_available():
                    #print('CUDA available')
                    cur_dev = torch.cuda.current_device()
                    #print('current device ' + str(cur_dev))
                    #print('device count ' + str(torch.cuda.device_count()))
                    #print('device name ' + torch.cuda.get_device_name(cur_dev))
                    input_tensor.cuda()
                    target_tensor.cuda()
                else:
                    print('CUDA not available')

                input_tensor, target_tensor = torch.from_numpy(scale_center_X).cuda(), torch.from_numpy(scale_center_Y).cuda()
                self.model.train_model(input_tensor, target_tensor, e)
                ## how to undo the scaling:
                #unscaled_X = utils.scale_and_center_reverse(scale_center_X, scale_params_low, mean_image_low, image_type = self.dataset.image_type)
                #unscaled_Y = utils.scale_and_center_reverse(scale_center_Y, scale_params_high, mean_image_high, image_type=self.dataset.image_type)

    def predict(self):
        #self.model.predict()

        # see self.dataset.X_val and self.dataset.Y_val
        pass

    def log_model(self):
        #self.logger.log(self.model)
        pass


def main():
    trainer = CNN_skipCo_trainer()
    trainer.fit()
    trainer.predict()
    trainer.log_model()


if __name__ == "__main__":
    main()
