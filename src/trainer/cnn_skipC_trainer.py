from data.data_loader import ProcessData
import trainer.utils as utils
from matplotlib import pyplot as plt
#import numpy as np

class CNN_skipCo_trainer(object):
    def __init__(self, image_type='OA'):
        self.image_type = image_type
        self.dataset = ProcessData(train_ratio=0.7,process_raw_data=False, image_type=self.image_type, get_scale_center=False)

        #self.model = CNN_skipCo(self.dataset)
        #self.logger = Logger(self)

    def fit(self):
        self.dataset.batch_names(batch_size=5) # call this to separate names into random batches
        # get scale and center parameters
        scale_params_low, scale_params_high = utils.load_params(image_type=self.image_type, param_type="scale_params")
        mean_image_low, mean_image_high = utils.load_params(image_type=self.image_type, param_type="mean_images")

        # in self.batch_number is the number of batches in the training set
        for i in range(self.dataset.batch_number):
            X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[i])
            print("The input batch:")
            print(X.shape)
            # scale and center the batch
        #    scale_center_X = utils.scale_and_center(X, scale_params_low, mean_image_low, image_type = self.image_type)
        #    scale_center_Y = utils.scale_and_center(Y, scale_params_high, mean_image_high, image_type=self.image_type)

            ## how to undo the scaling:
            #unscaled_X = utils.scale_and_center_reverse(scale_center_X, scale_params_low, mean_image_low, image_type = self.image_type)
            #unscaled_Y = utils.scale_and_center_reverse(scale_center_Y, scale_params_high, mean_image_high, image_type=self.image_type)
            #print(np.max(np.abs(unscaled_X - X))) # you have to import numpy first




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
