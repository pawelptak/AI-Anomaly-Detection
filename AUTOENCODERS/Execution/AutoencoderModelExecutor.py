from tensorflow.keras import Sequential

from Model.AutoEncoderModelBase import AutoEncoderModelBase


"""
The class allows to fit the model and perform the predictions
"""


class AutoencoderModelExecutor:
    def __init__(self, model: AutoEncoderModelBase, epochs=5):
        self.model = model.build_model()
        self.epochs = epochs

    def fit(self, X):
        self.model.fit(X, X, epochs=self.epochs)

    def predict(self, X):
        return self.model.predict(X)
