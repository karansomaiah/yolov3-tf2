import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Layer, Add


class RouteLayer(Layer):
    def __init__(self, indices=[], **kwargs):
        super(RouteLayer, self).__init__()
        self.indices=indices
        self.add = Add()

    def call(self, inputs):
        return self.add(inputs)
