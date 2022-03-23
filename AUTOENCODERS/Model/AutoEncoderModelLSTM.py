from operator import mod
from os import name
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from Model.AutoEncoderModelBase import AutoEncoderModelBase
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, RepeatVector
import numpy as np


class AutoEncoderModelLSTM(AutoEncoderModelBase):
    def __init__(self, features_number, window_size=20, units=[80, 50, 20]):
        self.features_number = features_number
        self.window_size = window_size
        self.units = units

    def build_model(self) -> Sequential:
        # encoder model with stacked LSTM
        encoder = Sequential(
            [
                LSTM(self.units[0], return_sequences=True, activation='selu', input_shape=(
                    self.window_size, self.features_number), dropout=0.2),
                LSTM(self.units[1], activation='selu', return_sequences=True),
                LSTM(self.units[2], activation='selu')], name='encoder')
        # decoder model with output dimension same as input dimension
        decoder = Sequential(
            [
                RepeatVector(self.window_size),
                LSTM(self.units[1], activation='selu', return_sequences=True),
                LSTM(self.units[0], activation='selu', return_sequences=True),
                TimeDistributed(Dense(self.features_number, activation='linear'))], name='decoder')
        # creating sequential autoencoder using encoder, decoder as layers
        autoencoder = Sequential([encoder, decoder], name='autoencoder')
        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.Huber(100.))

        encoder.summary()
        decoder.summary()
        autoencoder.summary()

        self.model = autoencoder
        return self.model


class AutoEncoderModelLSTM_2(AutoEncoderModelBase):
    def __init__(self, features_number, window_size, units):
        self.features_number = features_number
        self.window_size = window_size
        self.units = units

    def build_model(self) -> Sequential:
        # encoder model with stacked LSTM
        encoder = Sequential(
            [
                LSTM(self.units[0], return_sequences=True, activation='selu', input_shape=(
                    self.window_size, self.features_number), dropout=0.2),
                LSTM(self.units[1], activation='selu')], name='encoder')
        # decoder model with output dimension same as input dimension
        decoder = Sequential(
            [
                RepeatVector(self.window_size),
                LSTM(self.units[0], activation='selu', return_sequences=True),
                TimeDistributed(Dense(self.features_number, activation='linear'))], name='decoder')
        # creating sequential autoencoder using encoder, decoder as layers
        autoencoder = Sequential([encoder, decoder], name='autoencoder')
        autoencoder.compile(optimizer='adam', loss=tf.keras.losses.Huber(100.))

        encoder.summary()
        decoder.summary()
        autoencoder.summary()

        self.model = autoencoder
        return self.model


class AutoEncoderModelLSTM_3(AutoEncoderModelBase):
    def __init__(self, features_number, window_size, units):
        self.features_number = features_number
        self.window_size = window_size
        self.units = units

    def build_model(self) -> Sequential:
        # encoder model with stacked LSTM
        encoder = Sequential(
            [
                LSTM(self.units[0], return_sequences=True, activation='selu', input_shape=(
                    self.window_size, self.features_number), dropout=0.2),
                LSTM(self.units[1], activation='selu', return_sequences=True),
                LSTM(self.units[2], activation='selu')], name='encoder')
        # decoder model with output dimension same as input dimension
        decoder = Sequential(
            [
                RepeatVector(self.window_size),
                LSTM(self.units[1], activation='selu', return_sequences=True),
                LSTM(self.units[0], activation='selu', return_sequences=True),
                TimeDistributed(Dense(self.features_number, activation='linear'))], name='decoder')
        # creating sequential autoencoder using encoder, decoder as layers
        autoencoder = Sequential([encoder, decoder], name='autoencoder')
        autoencoder.compile(optimizer='sgd', loss=tf.keras.losses.Huber(100.))

        encoder.summary()
        decoder.summary()
        autoencoder.summary()

        self.model = autoencoder
        return self.model
