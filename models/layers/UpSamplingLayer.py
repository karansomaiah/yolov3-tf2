import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Layer, UpSampling1D, UpSampling2D, UpSampling3D


upsampling_map = {
    1: UpSampling1D,
    2: UpSampling2D,
    3: UpSampling3D
}


class UpSamplingLayer(Layer):
    def __init__(self,
                 factor=2):
        super(UpSamplingLayer, self).__init__()
        self.factor = factor
        self.upsample_layer = upsampling_map[self.factor]

    def call(self, inputs):
        return self.upsample_layer(inputs)
