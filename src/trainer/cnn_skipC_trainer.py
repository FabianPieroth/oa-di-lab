from data.data_loader import ProcessData


class CNN_skipCo_trainer(object):
    def __init__(self):
        self.dataset = ProcessData(train_ratio=0.3,process_raw_data=True, do_augment=True, image_type='US')

        #self.logger = Logger(self)

    def fit(self):
        self.dataset.batch_names(batch_size=5) # call this to separate names into random batches
        # in self.batch_number is the number of batches in the training set
        for i in range(self.dataset.batch_number):
            X, Y = self.dataset.create_train_batches(self.dataset.train_batch_chunks[i])
            print("The input batch:")
            print(X.shape)


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
