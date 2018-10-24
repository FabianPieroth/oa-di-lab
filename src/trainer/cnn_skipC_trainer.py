#import ...

class CNN_skipCo_trainer(object):
    def __init__(self):
        self.dataset = PreprocessImages()
        self.model = CNN_skipCo(self.dataset)
        self.logger = Logger(self)

    def fit(self):
        pass

    def predict(self):
        self.model.predict()

    def log_model(self):
        self.logger.log(self.model)


def main():
    trainer = CNN_skipCo_trainer()
    trainer.fit()
    trainer.predict()
    trainer.log_model()


if __name__ == "__main__":
    main()
