from data.data_loader import ProcessData
import trainer.utils as utils
import numpy as np


class CNN_skipCo_trainer(object):
    def __init__(self):
        self.dataset = ProcessData(train_ratio=0.3,process_raw_data=True, do_augment=True, image_type='US', get_scale_center=True)
        #self.logger = Logger(self)

    def fit(self):
        self.dataset.batch_names(batch_size=5) # call this to separate names into random batches
        # get scale and center parameters
        scale_params_low, scale_params_high = utils.load_params(image_type=self.dataset.image_type, param_type="scale_params")
        mean_image_low, mean_image_high = utils.load_params(image_type=self.dataset.image_type, param_type="mean_images")
        print(mean_image_low.shape)

        # in self.batch_number is the number of batches in the training set
        for i in range(self.dataset.batch_number):
            X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[i])
            print("The input batch:")
            print(X.shape)
            # scale and center the batch
            scale_center_X = utils.scale_and_center(X, scale_params_low, mean_image_low, image_type=self.dataset.image_type)
            scale_center_Y = utils.scale_and_center(Y, scale_params_high, mean_image_high, image_type=self.dataset.image_type)

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
