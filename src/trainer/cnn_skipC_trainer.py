from data.data_loader import ProcessData
import trainer.utils as utils
import numpy as np
from models import cnn_skipC_model


class CNN_skipCo_trainer(object):
    def __init__(self):
        self.dataset = ProcessData(train_ratio=0.3,process_raw_data=True, do_augment=False, image_type='US', get_scale_center=True)

        self.model = cnn_skipC_model.cnn_skipC_model()
        #self.logger = Logger(self)

    def fit(self):
        self.dataset.batch_names(batch_size=5) # call this to separate names into random batches
        # get scale and center parameters
        scale_params_low, scale_params_high = utils.load_params(image_type=self.dataset.image_type, param_type="scale_params")
        mean_image_low, mean_image_high = utils.load_params(image_type=self.dataset.image_type, param_type="mean_images")
        print(mean_image_low.shape)


        for e in range(0, epochs):
            # in self.batch_number is the number of batches in the training set
            for i in range(self.dataset.batch_number):
                X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[i])
                print("The input batch:")
                print(X.shape)
                # scale and center the batch
                scale_center_X = utils.scale_and_center(X, scale_params_low, mean_image_low, image_type=self.dataset.image_type)
                scale_center_Y = utils.scale_and_center(Y, scale_params_high, mean_image_high, image_type=self.dataset.image_type)
                # (N, H, W, C) to (N, C, H, W)
                scale_center_X = scale_center_X.reshape(scale_center_X.shape[0], scale_center_X[3], scale_center_X[1], scale_center_X[2])
                scale_center_Y = scale_center_Y.reshape(scale_center_Y.shape[0], scale_center_Y[3], scale_center_Y[1], scale_center_Y[2])
                self.model.train_model(scale_center_X, scale_center_Y)
                ## how to undo the scaling:
                #unscaled_X = utils.scale_and_center_reverse(scale_center_X, scale_params_low, mean_image_low, image_type = self.dataset.image_type)
                #unscaled_Y = utils.scale_and_center_reverse(scale_center_Y, scale_params_high, mean_image_high, image_type=self.dataset.image_type)

    epochs = 200
    net = AwesomeImageTranslator1000(learning_rate=0.001)

    print('start training')
    print('-----------------------------------------')
    for j in range(0, epochs):
        for i in range(0, len(input_batches)):
            batch = input_batches[i]
            target = target_batches[i]
            net.train_model(torch.from_numpy(np.array([batch]).reshape(batch_size, 1, 401, 401)),
                            torch.from_numpy(np.array([target]).reshape(batch_size, 1, 401, 401)), j)
    print('\n-----------------------------------------')
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
