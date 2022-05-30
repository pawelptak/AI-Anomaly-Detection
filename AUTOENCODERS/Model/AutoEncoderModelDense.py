from operator import mod
from os import name
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from Model.AutoEncoderModelBase import AutoEncoderModelBase
import numpy as np


"""
Autoencoder with 2 dense layers (1 for encoding, 1 for decoding)
"""


class AutoEncoderModelDense_1(AutoEncoderModelBase):
    def __init__(self, features_number, dense_size=20):
        self.features_number = features_number
        self.dense_size = dense_size

    def build_model(self) -> Sequential:
        self.model = Sequential([
            keras.Input(shape=(self.features_number,)),
            layers.Dense(np.round(self.dense_size),
                         activation='relu', name='encoder'),
            layers.Dense(self.features_number,
                         activation='relu', name='decoder'),
        ])

        self.model.summary()
        opt = optimizers.Adam(lr=0.001)
        self.model.compile(opt, loss='mse')

        return self.model


"""
Autoencoder with 4 dense layers (2 for encoding, 2 for decoding)
"""


class AutoEncoderModelDense_2(AutoEncoderModelBase):
    def __init__(self, features_number, dense_size_1, dense_size_2):
        self.features_number = features_number
        self.dense_size_1 = dense_size_1
        self.dense_size_2 = dense_size_2

    def build_model(self) -> Sequential:
        self.model = Sequential([
            keras.Input(shape=(self.features_number,)),
            layers.Dense(np.round(self.dense_size_1),
                         activation='relu'),
            layers.Dense(np.round(self.dense_size_2),
                         activation='relu'),
            layers.Dense(np.round(self.dense_size_1),
                         activation='relu'),
            layers.Dense(self.features_number, activation='relu'),
        ])

        self.model.summary()
        opt = optimizers.Adam(lr=0.001)
        self.model.compile(opt, loss='mse')

        return self.model
