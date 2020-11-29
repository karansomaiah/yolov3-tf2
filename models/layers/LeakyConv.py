import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU


class LeakyConvolution(Layer):
    def __init__(
        self, filters, kernel_size, strides, padding, data_format, activation, **kwargs
    ):
        super(LeakyConvolution, self).__init__("leaky_conv")
        self.filters = filters

        if type(kernel_size) == tuple:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)

        if type(strides) == tuple:
            self.strides = strides
        else:
            self.strides = (strides, strides)

        self.padding = padding
        self.data_format = data_format
        assert activation == None

        if len(kwargs) > 0:
            self.conv = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding="same",
                data_format="channels_last",
                activation=None,
                input_shape=kwargs["input_shape"],
            )
        else:
            self.conv = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding="same",
                data_format="channels_last",
                activation=None,
            )
        self.leaky_relu = LeakyReLU(alpha=0.3)

    # def build(self, input_shape):
    #     self.conv = Conv2D(
    #         filters=self.filters,
    #         kernel_size=self.kernel_size,
    #         strides=self.strides,
    #         padding="same",
    #         data_format="channels_last",
    #         activation=None,
    #     )
    #     self.leaky_relu = LeakyReLU(alpha=0.3)

    def call(self, input):
        x = self.conv(input)
        return self.leaky_relu(x)
