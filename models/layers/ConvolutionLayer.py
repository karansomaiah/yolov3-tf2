import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Layer, BatchNormalization

activation_fn_map = {
    0: None,  # NONE = 0;
    1: tf.keras.activations.relu,  # RELU = 1;
    2: tf.keras.activations.sigmoid,  # SIGMOID = 2;
    3: tf.keras.activations.hard_sigmoid,  # HARD_SIGMOID = 3;
    4: tf.keras.activations.swish,  # SWISH = 4;
    5: tf.keras.activations.softmax,  # SOFTMAX = 5;
    6: tf.keras.activations.softplus,  # SOFTPLUS = 6;
    7: tf.keras.activations.softsign,  # SOFTSIGN = 7;
    8: tf.keras.activations.tanh,  # TANH = 8;
    9: tf.keras.activations.selu,  # SELU = 9;
    10: tf.keras.activations.elu,  # ELU = 10;
    11: tf.keras.activations.exponential,  # EXPONENT = 11;
    12: tf.keras.layers.LeakyReLU(),  # LEAKY = 12;
}


class ConvolutionLayer(Layer):
    """Convolution Layer implementation"""

    def __init__(
        self,
        num_filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation_int=None,
        **kwargs
    ):
        super(ConvolutionLayer, self).__init__()
        self.num_filters = num_filters
        if type(kernel_size) == tuple:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)
        if type(strides) == tuple:
            self.strides = strides
        else:
            self.strides = (strides, strides)
        self.convolution_layer = Conv2D(
            self.num_filters,
            self.kernel_size,
            self.strides,
            padding="same",
            data_format="channels_last",
        )
        self.bn = BatchNormalization
        self.activation_fn = activation_fn_map[activation_int]

    def call(self, inputs):
        return self.activation_fn(self.convolution_layer(inputs))
