import time
import numpy as np

import tensorflow as tf


class Combine_CNN_RNN(tf.keras.Model):
    def __init__(self, num_class, units, dropout, advance=True, include_auto=True):
        super(Combine_CNN_RNN, self).__init__()
        self.units = units
        self.dropout = dropout
        self.num_class = num_class
        self.include_auto = include_auto

        if advance:
            self.rnn = tf.keras.layers.LSTM(self.units*4, activation='tanh')
            self.cnn = tf.keras.models.Sequential([
                                                   tf.keras.layers.Conv2D(3, 1, padding="same"),
                                                   tf.keras.applications.ResNet50(include_top=False),
                                                   tf.keras.layers.Flatten(),
                                                   tf.keras.layers.Dense(512),
                                                   tf.keras.layers.Dense(128),
                                                   tf.keras.layers.Dense(32)
                                                   ])
        else:
            self.auto_rnn = tf.keras.layers.GRU(self.units*16, activation='tanh')
            self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.units*8, activation='tanh'))
            self.cnn = tf.keras.models.Sequential([
                                                   tf.keras.layers.Conv2D(self.units,   3, activation='relu', padding="same"),
                                                   tf.keras.layers.MaxPooling2D(),
                                                   tf.keras.layers.Conv2D(self.units*2, 3, activation='relu', padding="same"),
                                                   tf.keras.layers.MaxPooling2D(),
                                                   tf.keras.layers.Conv2D(self.units*4, 3, activation='relu', padding="same"),
                                                   tf.keras.layers.MaxPooling2D(),
                                                   tf.keras.layers.Conv2D(self.units*8, 3, activation='relu', padding="same"),
                                                   tf.keras.layers.Dropout(0.25),
                                                   tf.keras.layers.Flatten(),
                                                   tf.keras.layers.Dense(128)
                                                   ])

        # apply cnn for each spectrogram slice then pass the features to rnn
        self.time_dist = tf.keras.layers.TimeDistributed(self.cnn)
        self.atten = tf.keras.layers.Attention()

        self.dropout = tf.keras.layers.Dropout(self.dropout)
        self.concat = tf.keras.layers.Concatenate()
        self.fc = tf.keras.models.Sequential([
                                              tf.keras.layers.Dense(128),
                                              tf.keras.layers.Dense(self.num_class, activation='softmax')
                                              ])


def call(self, *input):
    spec = input[0]
    auto = input[1]

    spec = self.time_dist(spec)
    spec = self.rnn(spec)

    if self.include_auto:
        auto = self.auto_rnn(auto)
        x = self.atten([spec, auto])
        x = self.concat([spec, x])

    else:
        x = spec

    x = self.dropout(x)
    x = self.fc(x)
    return x
