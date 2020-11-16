import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
from collections import OrderedDict, namedtuple
from layers.LeakyConv import LeakyConvolution
from layers.YoloLayer import YoloLayer

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    LeakyReLU,
    Add,
    Input,
    UpSampling2D,
    AveragePooling2D,
)
from tensorflow.keras import Model
from utils.network_placeholders import (
    Convolution,
    Connection,
    Residual,
    Route,
    UpSample,
    Yolo
)


class Backbone(keras.Model):
    def __init__(self, parsed_config):
        super(Backbone, self).__init__()
        self.build()

    def call(self, input):
        pass

    def 


# class Backbone(keras.Model):
#     """Creates a backbone given a config file."""

#     def __init__(self, parsed_config, convolution_activation, **kwargs):
#         super(Backbone, self).__init__(name, **kwargs)
#         self.network_definition = parsed_config
#         self.network_mapping = {
#             Convolution: Conv2D,
#             Connection: Add,
#             UpSample: UpSampling2D,
#             Yolo: YoloLayer
#         }
#         self.conv_activation = convolution_activation
#         self.layers = OrderedDict()
#         self.create_layers()

#     def call(self, inputs):
#         return

#     def create_layers(self):
#         """
#             Will parse the attributes from the network definition and create a 
#             Model from it. 

#             @todo :
#                 - Make this cleaner by calling one function that is able this cleanly.
#         """
#         LayerTuple = namedtuple('Layer', ['call', 'args'])
#         for layer_index, layer_def in self.network_definition:

#             if type(layer_def) == Residual:
#                 continue # Do nothing about it. The layers have already been expanded.
#             elif type(layer_def) == Convolution:
#                 if self.conv_activation == "lrelu":
#                     l = LayerTuple(LeakyConvolution(filters=layer_def.num_kernels,
#                                                 kernel_size=(layer_def.kernel_h,
#                                                              layer_def.kernel_w),
#                                                 strides=(layer_def.stride_h,
#                                                          layer_def.stride_w),
#                                                 padding="same",
#                                                 data_format="channels_last",
#                                                 activation=None), "")
#                     self.layers[layer_index]  = l
#                 else:
#                     l = LayerTuple(self.network_mapping[Convolution](filters=layer_def.num_kernels, 
#                                                     kernel_size=(layer_def.kernel_h, layer_def.kernel_w),
#                                                     strides=(layer_def.stride_h, layer_def.stride_w),
#                                                     padding="same",
#                                                     data_format="channels_last",
#                                                     activation=self.conv_activation), "")
#                     self.layers[layer_index] = l
#             elif type(layer_def) == Yolo:
#                 l = LayerTuple(self.network_mapping[Yolo](layer_def.anchors, 1), "")
#                 self.layers[layer_index] = l
#             elif type(layer_def) == UpSample:
#                 l = LayerTuple(self.network_mapping[UpSample](size=(layer_def.factor, layer_def.factor)))
#             elif type(layer_def) == list:
#                 if type(layer_def[0]) == Connection:
#                     indices = layer_def[1:]
#                     l = LayerTuple(Add(), indices)
#                     self.layers[layer_index] = l
#                 elif type(layer_def[0]) == Route:
#                     indices = 
#                     l = LayerTuple('', )
#                     pass
#                 else:
#                     print("[ERROR]: Error creating layer.")
#             else:
#                 pass
#                 # Do Nothing
#         print("Layers are set!")
