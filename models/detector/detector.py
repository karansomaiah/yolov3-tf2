import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

from collections import OrderedDict
from utils.utils import *
from utils import config_parser, network_placeholders

from ..layers.LeakyConv import LeakyConvolution
from ..layers.YoloLayer import YoloLayer

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    LeakyReLU,
    Add,
    Concatenate,
    Input,
    AveragePooling2D,
    UpSampling2D,
)
from tensorflow.keras import Model


class Detector(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, parsed_layers, num_classes, **kwargs):
        super(Detector, self).__init__()
        self.num_classes = num_classes
        self.parsed_config = parsed_layers
        self.layer_outputs = OrderedDict()
        self.layers_list = []
        self.anchor_indices = []

    def call(self, inputs):
        prev = None
        count = 0
        for layer_tuple in self.layers_list:
            if layer_tuple.args != DEFAULT_LAYER_ARG:
                if layer_tuple.call == "":  # Route layer (with only 1 arg)
                    out = self.layer_outputs[layer_tuple.args[0]]
                    self.layer_outputs[layer_tuple.index] = out
                    prev = out
                else:  # Add layer from Connection or Route layer
                    out = layer_tuple.call(
                        [self.layer_outputs[index] for index in layer_tuple.args]
                    )
                    self.layer_outputs[layer_tuple.index] = out
                    prev = out
            else:
                if prev == None and len(self.layer_outputs) == 0:  # 1st layer handling
                    out = layer_tuple.call(inputs)
                    self.layer_outputs[layer_tuple.index] = out
                    prev = out
                else:
                    out = layer_tuple.call(prev)
                    self.layer_outputs[layer_tuple.index] = out
                    prev = out
            # print(layer_tuple.index, out.shape)
        return [self.layer_outputs[index] for index in self.anchor_indices]

    def build_model(self, input_shape):
        model_input = Input(shape=input_shape)

        # assumption is the input_shape is of style
        # [N, H, W, C] because Tensorflow “¯\_(ツ)_/¯“
        for parsed_layer_index, parsed_layer in self.parsed_config:
            if parsed_layer_index != IGNORE_LAYER_INDEX:
                # Convolution
                if type(parsed_layer) == config_parser.Convolution:
                    # print("In Convolution")
                    if parsed_layer_index == 1:
                        layer_definition = LayerTuple(
                            LeakyConvolution(
                                filters=parsed_layer.attributes["num_kernels"],
                                kernel_size=(
                                    parsed_layer.attributes["kernel_h"],
                                    parsed_layer.attributes["kernel_w"],
                                ),
                                strides=(
                                    parsed_layer.attributes["stride_h"],
                                    parsed_layer.attributes["stride_w"],
                                ),
                                padding="same",
                                data_format="channels_last",
                                activation=None,
                                # input_shape=self.ip_shape,
                            ),
                            DEFAULT_LAYER_ARG,
                            parsed_layer_index,
                        )
                        self.layers_list.append(layer_definition)
                        # activation_definition = LayerTuple(
                        #     LeakyReLU(alpha=0.3), DEFAULT_LAYER_ARG, parsed_layer_index
                        # )
                        # self.layers_list.append(activation_definition)
                    else:
                        layer_definition = LayerTuple(
                            LeakyConvolution(
                                filters=parsed_layer.attributes["num_kernels"],
                                kernel_size=(
                                    parsed_layer.attributes["kernel_h"],
                                    parsed_layer.attributes["kernel_w"],
                                ),
                                strides=(
                                    parsed_layer.attributes["stride_h"],
                                    parsed_layer.attributes["stride_w"],
                                ),
                                padding="same",
                                data_format="channels_last",
                                activation=None,
                            ),
                            DEFAULT_LAYER_ARG,
                            parsed_layer_index,
                        )
                        self.layers_list.append(layer_definition)
                        # activation_definition = LayerTuple(
                        #     LeakyReLU(alpha=0.3), DEFAULT_LAYER_ARG, parsed_layer_index
                        # )
                        # self.layers_list.append(activation_definition)

                # UpSample Layer
                elif type(parsed_layer) == config_parser.UpSample:
                    # print("In UpSample")
                    layer_definition = LayerTuple(
                        UpSampling2D(
                            size=(
                                parsed_layer.attributes["factor"],
                                parsed_layer.attributes["factor"],
                            ),
                            data_format="channels_last",
                            interpolation="bilinear",
                        ),
                        DEFAULT_LAYER_ARG,
                        parsed_layer_index,
                    )
                    self.layers_list.append(layer_definition)
                # Yolo Layer
                elif type(parsed_layer) == config_parser.Yolo:
                    # print("In Yolo")
                    layer_definition = LayerTuple(
                        YoloLayer(parsed_layer.attributes["anchors"], self.num_classes),
                        DEFAULT_LAYER_ARG,
                        parsed_layer_index,
                    )
                    self.layers_list.append(layer_definition)
                    self.anchor_indices.append(parsed_layer_index)
                # Connection
                else:
                    try:
                        if type(parsed_layer[0]) == config_parser.Connection:
                            layer_args = parsed_layer[1:]
                            layer_definition = LayerTuple(
                                Add(), layer_args, parsed_layer_index
                            )
                            self.layers_list.append(layer_definition)
                        if type(parsed_layer[0]) == config_parser.Route:
                            layer_args = parsed_layer[1:]
                            if len(layer_args) > 1:
                                layer_definition = LayerTuple(
                                    Concatenate(), layer_args, parsed_layer_index
                                )
                            else:
                                layer_definition = LayerTuple(
                                    "", layer_args, parsed_layer_index
                                )
                            self.layers_list.append(layer_definition)
                    except:
                        print("[ERROR]: Some weird layer")

        return Model(inputs=[model_input], outputs=self.call(model_input))