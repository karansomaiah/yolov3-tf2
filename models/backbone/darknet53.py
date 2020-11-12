#! python3
""" Code holding the darknet53 backbone for YOLOV3
"""

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


class DarkNet53(Model):
    def __init__(self, num_classes):
        super(DarkNet53, self).__init__()
        self.num_classes = num_classes
        self.conv_blocks = {
            # block 1 (after first two convolutional inputs)
            "block1": {
                "input": "conv2",
                "design": [(32, 1, 1, 1), (64, 3, 1, 1)],
                "residual": True,
            },
            # block 2
            "block2": {
                "input": "conv3",
                "design": [(64, 1, 1, 1), (128, 3, 1, 1)] * 2,
                "residual": True,
            },
            # block 3
            "block3": {
                "input": "conv4",
                "design": [(128, 1, 1, 1), (256, 3, 1, 1)] * 8,
                "residual": True,
            },
            # block 4
            "block4": {
                "input": "conv5",
                "design": [(256, 1, 1, 1), (512, 3, 1, 1)] * 8,
                "residual": True,
            },
            # block 5
            "block5": {
                "input": "conv6",
                "design": [(512, 1, 1, 1), (1024, 3, 1, 1)] * 4,
                "residual": True,
            },
        }

        self.conv1 = Conv2D(
            32,
            3,
            (2, 2),
            padding="same",
            data_format="channels_last",
            activation=None,
            name="conv1",
        )
        self.conv2 = Conv2D(
            64,
            3,
            (2, 2),
            padding="same",
            data_format="channels_last",
            activation=None,
            name="conv2",
        )
        self.block1 = self.build_blocks("block1", self.conv_blocks["block1"])

        self.conv3 = Conv2D(
            128,
            3,
            (2, 2),
            padding="same",
            data_format="channels_last",
            activation=None,
            name="conv3",
        )
        self.block2 = self.build_blocks("block2", self.conv_blocks["block2"])

        self.conv4 = Conv2D(
            256,
            3,
            (2, 2),
            padding="same",
            data_format="channels_last",
            activation=None,
            name="conv4",
        )
        self.block3 = self.build_blocks("block3", self.conv_blocks["block3"])

        self.conv5 = Conv2D(
            512,
            3,
            (2, 2),
            padding="same",
            data_format="channels_last",
            activation=None,
            name="conv5",
        )
        self.block4 = self.build_blocks("block4", self.conv_blocks["block4"])

        self.conv6 = Conv2D(
            1024,
            3,
            (2, 2),
            padding="same",
            data_format="channels_last",
            activation=None,
            name="conv6",
        )
        self.block5 = self.build_blocks("block5", self.conv_blocks["block5"])
        self.flatten = Flatten()
        self.feature_outputs = []

    def call(self, x):
        x = self.conv1(x)
        x = LeakyReLU(alpha=0.3)(x)

        x = self.conv2(x)
        conv2 = LeakyReLU(alpha=0.3)(x)
        x = self.block1[0](conv2)
        for BlockConvLayer in self.block1[1:]:
            x = BlockConvLayer(x)
        x = Add()([conv2, x])

        x = self.conv3(x)
        conv3 = LeakyReLU(alpha=0.3)(x)
        x = self.block2[0](conv3)
        for BlockConvLayer in self.block2[1:]:
            x = BlockConvLayer(x)
        x = Add()([conv3, x])

        x = self.conv4(x)
        conv4 = LeakyReLU(alpha=0.3)(x)
        x = self.block3[0](conv4)
        for BlockConvLayer in self.block3[1:]:
            x = BlockConvLayer(x)
        x = Add()([conv4, x])

        x = self.conv5(x)
        conv5 = LeakyReLU(alpha=0.3)(x)
        x = self.block4[0](conv5)
        for BlockConvLayer in self.block4[1:]:
            x = BlockConvLayer(x)
        x = Add()([conv5, x])

        x = self.conv6(x)
        conv6 = LeakyReLU(alpha=0.3)(x)
        x = self.block5[0](conv6)
        for BlockConvLayer in self.block5[1:]:
            x = BlockConvLayer(x)
        x = Add()([conv6, x])

        x = AveragePooling2D()(x)
        x = self.flatten(x)
        x = Dense(1024, activation="relu")(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dense(self.num_classes)(x)
        return x

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)

    # def build_blocks(self, block_name, block):
    #     """Builds a skip-connection inspired DarkNet block.

    #     Args:
    #         block (str, dict): Dictionary holding block information
    #         Example:
    #         {
    #             "input": "conv5",
    #             "design": [(256, 1, 1, 1), (512, 3, 3, 1)] * 8,
    #             "residual": True,
    #         }
    #     """
    #     block_prefix = str(block_name)
    #     residual_blocks = []
    #     num_block_convs = len(block["design"])

    #     for block_num in range(num_block_convs):
    #         conv_name = block_prefix + "_conv_" + str(block_num + 1)
    #         conv_filters, kernel_size, height_stride, width_stride = block["design"][
    #             block_num
    #         ]
    #         residual_blocks.append(
    #             keras.layers.Conv2D(
    #                 conv_filters,
    #                 kernel_size,
    #                 (height_stride, width_stride),
    #                 padding="same",
    #                 data_format="channels_last",
    #                 activation=None,
    #                 name=conv_name,
    #             )
    #         )
    #         residual_blocks.append(
    #             keras.layers.LeakyReLU(alpha=0.3, name=conv_name + "_leaky")
    #         )

    #     return residual_blocks

    def residual_blocks(input, config, output):
