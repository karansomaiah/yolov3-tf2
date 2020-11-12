import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    LeakyReLU,
    Add,
    Input,
    AveragePooling2D,
)
from tensorflow.keras import Model


class Detector(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, **kwargs):
        super(Detector, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return
