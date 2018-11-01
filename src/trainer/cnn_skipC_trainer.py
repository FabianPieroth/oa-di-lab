from data.data_loader import ProcessData

class CNN_skipCo_trainer(object):
    def __init__(self):
        self.dataset = ProcessData(train_ratio=0.3,process_raw_data=True)
        #self.model = CNN_skipCo(self.dataset)
        #self.logger = Logger(self)

    def fit(self):
        pass

    def predict(self):
        #self.model.predict()
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
