#!/usr/bin/env python
"""
"""

# regular imports
# from collections import OrderedDict

# tensorflow imports
import tensorflow as tf

# import models.layer_builder as layer_builder
from models.layers import (
    ConvolutionLayer,
    ResidualBlock,
    RouteLayer,
    UpSamplingLayer,
    YoloLayer,
)
from tensorflow.keras.layers import Add, Concatenate


class Detector(tf.keras.Model):
    """Detector - tf.keras.Model.
    This model class shall accept inputs, outputs, that are existing keras
    layers already defined in separate code.

    This class definition shall redefine the following abstract functions:
        - compile
        - train_step
        - test_step
    """

    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(**kwargs)

    def compile(self, loss_fn1, loss_fn2, loss_fn3, **kwargs):
        super().compile(**kwargs)
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.loss_fn3 = loss_fn3

    def train_step(self, data):
        images = data[0]
        labels = data[1]
        with tf.GradientTape() as tape:
            # get outputs
            outputs = self(images, training=True)
            output1 = outputs[0]
            output2 = outputs[1]
            output3 = outputs[2]
            # calculate loss
            loss1 = self.loss_fn1(labels, output1)
            loss2 = self.loss_fn2(labels, output2)
            loss3 = self.loss_fn3(labels, output3)
            loss = loss1 + loss2 + loss3
            # apply gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return {"loss": loss}

    def test_step(self, data):
        images, labels = data
        # get outputs
        outputs = self(images, training=True)
        output1 = outputs[0]
        output2 = outputs[1]
        output3 = outputs[2]
        # calculate loss
        loss1 = self.loss_fn1(labels, output1)
        loss2 = self.loss_fn2(labels, output2)
        loss3 = self.loss_fn3(labels, output3)
        loss = loss1 + loss2 + loss3
        return {"loss": loss}


def build(model_config, input_shape, batch_size):
    inputs = []
    outputs = []
    anchors = []

    # input
    model_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

    # # ------------------- model_def ------------------------ #
    # conv1
    conv1 = ConvolutionLayer.ConvolutionLayer(32, (3, 3), (1, 1), 12, name="conv1")(
        model_input
    )
    # conv2
    conv2 = ConvolutionLayer.ConvolutionLayer(64, (3, 3), (2, 2), 12, name="conv2")(
        conv1
    )

    # residual block 1: two convs @ 1
    # res1 conv1
    res1_conv1 = ConvolutionLayer.ConvolutionLayer(
        32, (1, 1), (1, 1), 12, name="res1_conv1"
    )(conv2)
    # res1 conv2
    res1_conv2 = ConvolutionLayer.ConvolutionLayer(
        64, (3, 3), (1, 1), 12, name="res1_conv2"
    )(res1_conv1)
    # res1 out
    res1_out = Add(name="res1_out")([conv2, res1_conv2])

    # conv3
    conv3 = ConvolutionLayer.ConvolutionLayer(128, (3, 3), (2, 2), 12, name="conv3")(
        res1_out
    )

    # residual block 2: two convs @ 2
    # res2 conv1
    res2_conv1 = ConvolutionLayer.ConvolutionLayer(
        64, (1, 1), (1, 1), 12, name="res2_conv1"
    )(conv3)
    # res2 conv2
    res2_conv2 = ConvolutionLayer.ConvolutionLayer(
        128, (3, 3), (1, 1), 12, name="res2_conv2"
    )(res2_conv1)
    # res2 interm1
    res2_interm1 = Add()([conv3, res2_conv2])
    # res2 conv3
    res2_conv3 = ConvolutionLayer.ConvolutionLayer(
        64, (1, 1), (1, 1), 12, name="res2_conv3"
    )(res2_interm1)
    # res2 conv4
    res2_conv4 = ConvolutionLayer.ConvolutionLayer(
        128, (3, 3), (1, 1), 12, name="res2_conv4"
    )(res2_conv3)
    # res2 out
    res2_out = Add(name="res2_out")([res2_interm1, res2_conv4])

    # conv4
    conv4 = ConvolutionLayer.ConvolutionLayer(256, (3, 3), (2, 2), 12, name="conv4")(
        res2_out
    )

    # residual block 3: two convs @ 8
    # res3 conv1
    res3_conv1 = ConvolutionLayer.ConvolutionLayer(
        128, (1, 1), (1, 1), 12, name="res3_conv1"
    )(conv4)
    # res3 conv2
    res3_conv2 = ConvolutionLayer.ConvolutionLayer(
        256, (3, 3), (1, 1), 12, name="res3_conv2"
    )(res3_conv1)
    # res3 interm1
    res3_interm1 = Add(name="res3_interm1")([conv4, res3_conv2])
    # res3 conv3
    res3_conv3 = ConvolutionLayer.ConvolutionLayer(
        128, (1, 1), (1, 1), 12, name="res3_conv3"
    )(res3_interm1)
    # res3 conv4
    res3_conv4 = ConvolutionLayer.ConvolutionLayer(
        256, (3, 3), (1, 1), 12, name="res3_conv4"
    )(res3_conv3)
    # res3 interm2
    res3_interm2 = Add(name="res3_interm2")([res3_interm1, res3_conv4])
    # res3 conv5
    res3_conv5 = ConvolutionLayer.ConvolutionLayer(
        128, (1, 1), (1, 1), 12, name="res3_conv5"
    )(res3_interm2)
    # res3 conv6
    res3_conv6 = ConvolutionLayer.ConvolutionLayer(
        256, (3, 3), (1, 1), 12, name="res3_conv6"
    )(res3_conv5)
    # res3 interm3
    res3_interm3 = Add(name="res3_interm3")([res3_interm2, res3_conv6])
    # res3 conv7
    res3_conv7 = ConvolutionLayer.ConvolutionLayer(
        128, (1, 1), (1, 1), 12, name="res3_conv7"
    )(res3_interm3)
    # res3 conv8
    res3_conv8 = ConvolutionLayer.ConvolutionLayer(
        256, (3, 3), (1, 1), 12, name="res3_conv8"
    )(res3_conv7)
    # res3 interm4
    res3_interm4 = Add(name="res3_interm4")([res3_interm3, res3_conv8])
    # res3 conv9
    res3_conv9 = ConvolutionLayer.ConvolutionLayer(
        128, (1, 1), (1, 1), 12, name="res3_conv9"
    )(res3_interm4)
    # res3 conv10
    res3_conv10 = ConvolutionLayer.ConvolutionLayer(
        256, (3, 3), (1, 1), 12, name="res3_conv10"
    )(res3_conv9)
    # res3 interm5
    res3_interm5 = Add(name="res3_interm5")([res3_interm4, res3_conv10])
    # res3 conv11
    res3_conv11 = ConvolutionLayer.ConvolutionLayer(
        128, (1, 1), (1, 1), 12, name="res3_conv11"
    )(res3_interm5)
    # res3 conv12
    res3_conv12 = ConvolutionLayer.ConvolutionLayer(
        256, (3, 3), (1, 1), 12, name="res3_conv12"
    )(res3_conv11)
    # res3 interm6
    res3_interm6 = Add(name="res3_interm6")([res3_interm5, res3_conv12])
    # res3 conv13
    res3_conv13 = ConvolutionLayer.ConvolutionLayer(
        128, (1, 1), (1, 1), 12, name="res3_conv13"
    )(res3_interm6)
    # res3 conv14
    res3_conv14 = ConvolutionLayer.ConvolutionLayer(
        256, (3, 3), (1, 1), 12, name="res3_conv14"
    )(res3_conv13)
    # res3 interm7
    res3_interm7 = Add(name="res3_interm7")([res3_interm6, res3_conv14])
    # res3 conv15
    res3_conv15 = ConvolutionLayer.ConvolutionLayer(
        128, (1, 1), (1, 1), 12, name="res3_conv15"
    )(res3_interm7)
    # res3 conv16
    res3_conv16 = ConvolutionLayer.ConvolutionLayer(
        256, (3, 3), (1, 1), 12, name="res3_conv16"
    )(res3_conv15)
    # res3 out
    res3_out = Add(name="res3_out")([res3_interm7, res3_conv16])

    # conv5
    conv5 = ConvolutionLayer.ConvolutionLayer(512, (3, 3), (2, 2), 12, name="conv5")(
        res3_out
    )

    # residual block 4: two convs @ 8
    # res4 conv1
    res4_conv1 = ConvolutionLayer.ConvolutionLayer(
        256, (1, 1), (1, 1), 12, name="res4_conv1"
    )(conv5)
    # res4 conv2
    res4_conv2 = ConvolutionLayer.ConvolutionLayer(
        512, (3, 3), (1, 1), 12, name="res4_conv2"
    )(res4_conv1)
    # res4 interm1
    res4_interm1 = Add(name="res4_interm1")([conv5, res4_conv2])
    # res4 conv3
    res4_conv3 = ConvolutionLayer.ConvolutionLayer(
        256, (1, 1), (1, 1), 12, name="res4_conv3"
    )(res4_interm1)
    # res4 conv4
    res4_conv4 = ConvolutionLayer.ConvolutionLayer(
        512, (3, 3), (1, 1), 12, name="res4_conv4"
    )(res4_conv3)
    # res4 interm2
    res4_interm2 = Add(name="res4_interm2")([res4_interm1, res4_conv4])
    # res4 conv5
    res4_conv5 = ConvolutionLayer.ConvolutionLayer(
        256, (1, 1), (1, 1), 12, name="res4_conv5"
    )(res4_interm2)
    # res4 conv6
    res4_conv6 = ConvolutionLayer.ConvolutionLayer(
        512, (3, 3), (1, 1), 12, name="res4_conv6"
    )(res4_conv5)
    # res4 interm3
    res4_interm3 = Add(name="res4_interm3")([res4_interm2, res4_conv6])
    # res4 conv7
    res4_conv7 = ConvolutionLayer.ConvolutionLayer(
        256, (1, 1), (1, 1), 12, name="res4_conv7"
    )(res4_interm3)
    # res4 conv8
    res4_conv8 = ConvolutionLayer.ConvolutionLayer(
        512, (3, 3), (1, 1), 12, name="res4_conv8"
    )(res4_conv7)
    # res4 interm4
    res4_interm4 = Add(name="res4_interm4")([res4_interm3, res4_conv8])
    # res4 conv9
    res4_conv9 = ConvolutionLayer.ConvolutionLayer(
        256, (1, 1), (1, 1), 12, name="res4_conv9"
    )(res4_interm4)
    # res4 conv10
    res4_conv10 = ConvolutionLayer.ConvolutionLayer(
        512, (3, 3), (1, 1), 12, name="res4_conv10"
    )(res4_conv9)
    # res4 interm5
    res4_interm5 = Add(name="res4_interm5")([res4_interm4, res4_conv10])
    # res4 conv11
    res4_conv11 = ConvolutionLayer.ConvolutionLayer(
        256, (1, 1), (1, 1), 12, name="res4_conv11"
    )(res4_interm5)
    # res4 conv12
    res4_conv12 = ConvolutionLayer.ConvolutionLayer(
        512, (3, 3), (1, 1), 12, name="res4_conv12"
    )(res4_conv11)
    # res4 interm6
    res4_interm6 = Add(name="res4_interm6")([res4_interm5, res4_conv12])
    # res4 conv13
    res4_conv13 = ConvolutionLayer.ConvolutionLayer(
        256, (1, 1), (1, 1), 12, name="res4_conv13"
    )(res4_interm6)
    # res4 conv14
    res4_conv14 = ConvolutionLayer.ConvolutionLayer(
        512, (3, 3), (1, 1), 12, name="res4_conv14"
    )(res4_conv13)
    # res4 interm7
    res4_interm7 = Add(name="res4_interm7")([res4_interm6, res4_conv14])
    # res4 conv15
    res4_conv15 = ConvolutionLayer.ConvolutionLayer(
        256, (1, 1), (1, 1), 12, name="res4_conv15"
    )(res4_interm7)
    # res4 conv16
    res4_conv16 = ConvolutionLayer.ConvolutionLayer(
        512, (3, 3), (1, 1), 12, name="res4_conv16"
    )(res4_conv15)
    # res4 out
    res4_out = Add(name="res4_out")([res4_interm7, res4_conv16])

    # conv6
    conv6 = ConvolutionLayer.ConvolutionLayer(1024, (3, 3), (2, 2), 12, name="conv6")(
        res4_out
    )
    # residual block 5: two convs @ 4
    # res5_conv1
    res5_conv1 = ConvolutionLayer.ConvolutionLayer(
        512, (1, 1), (1, 1), 12, name="res5_conv1"
    )(conv6)
    # res5_conv2
    res5_conv2 = ConvolutionLayer.ConvolutionLayer(
        1024, (3, 3), (1, 1), 12, name="res5_conv2"
    )(res5_conv1)
    # res5_interm1
    res5_interm1 = Add(name="res5_interm1")([conv6, res5_conv2])
    # res5_conv3
    res5_conv3 = ConvolutionLayer.ConvolutionLayer(
        512, (1, 1), (1, 1), 12, name="res5_conv3"
    )(res5_interm1)
    # res5_conv4
    res5_conv4 = ConvolutionLayer.ConvolutionLayer(
        1024, (3, 3), (1, 1), 12, name="res5_conv4"
    )(res5_conv3)
    # res5_interm2
    res5_interm2 = Add(name="res5_interm2")([res5_interm1, res5_conv4])
    # res5_conv5
    res5_conv5 = ConvolutionLayer.ConvolutionLayer(
        512, (1, 1), (1, 1), 12, name="res5_conv5"
    )(res5_interm2)
    # res5_conv6
    res5_conv6 = ConvolutionLayer.ConvolutionLayer(
        1024, (3, 3), (1, 1), 12, name="res5_conv6"
    )(res5_conv5)
    # res5_interm3
    res5_interm3 = Add(name="res5_interm3")([res5_interm2, res5_conv6])
    # res5_conv7
    res5_conv7 = ConvolutionLayer.ConvolutionLayer(
        512, (1, 1), (1, 1), 12, name="res5_conv7"
    )(res5_interm3)
    # res5_conv8
    res5_conv8 = ConvolutionLayer.ConvolutionLayer(
        1024, (3, 3), (1, 1), 12, name="res5_conv8"
    )(res5_conv7)
    # res5_out
    res5_out = Add(name="res5_out")([res5_interm3, res5_conv8])

    # convolutions leading upto yolo1
    # conv7
    conv7 = ConvolutionLayer.ConvolutionLayer(512, (1, 1), (1, 1), 12, name="conv7")(
        res5_out
    )
    # conv8
    conv8 = ConvolutionLayer.ConvolutionLayer(1024, (3, 3), (1, 1), 12, name="conv8")(
        conv7
    )
    # conv9
    conv9 = ConvolutionLayer.ConvolutionLayer(512, (1, 1), (1, 1), 12, name="conv9")(
        conv8
    )
    # conv10
    conv10 = ConvolutionLayer.ConvolutionLayer(1024, (3, 3), (1, 1), 12, name="conv10")(
        conv9
    )
    # conv11
    conv11 = ConvolutionLayer.ConvolutionLayer(512, (1, 1), (1, 1), 12, name="conv11")(
        conv10
    )
    # conv12
    conv12 = ConvolutionLayer.ConvolutionLayer(1024, (3, 3), (1, 1), 12, name="conv12")(
        conv11
    )
    # YOLO LAYER 1
    yolo1_anchors = [[116, 90], [156, 198], [373, 326]]
    yolo1 = YoloLayer.YoloLayer(
        anchors=yolo1_anchors,
        num_classes=1,
        num=3,
        jitter=0.3,
        ignore_threshold=0.7,
        truth_threshold=1.0,
        random=1,
        name="yolo1",
    )(conv12)

    # YOLO LAYER 2
    # conv13
    conv13 = ConvolutionLayer.ConvolutionLayer(256, (1, 1), (1, 1), 12, name="conv13")(
        conv11
    )
    # upsample conv13
    upsample_conv13 = UpSamplingLayer.UpSamplingLayer(factor=2, name="upsample_conv13")(
        conv13
    )
    # route i.e Concatenate
    concat_upconv13_res4out = Concatenate()([upsample_conv13, res4_out])
    # conv 14
    conv14 = ConvolutionLayer.ConvolutionLayer(256, (1, 1), (1, 1), 12, name="conv14")(
        concat_upconv13_res4out
    )
    conv15 = ConvolutionLayer.ConvolutionLayer(512, (3, 3), (1, 1), 12, name="conv15")(
        conv14
    )
    conv16 = ConvolutionLayer.ConvolutionLayer(256, (1, 1), (1, 1), 12, name="conv16")(
        conv15
    )
    conv17 = ConvolutionLayer.ConvolutionLayer(512, (3, 3), (1, 1), 12, name="conv17")(
        conv16
    )
    conv18 = ConvolutionLayer.ConvolutionLayer(256, (1, 1), (1, 1), 12, name="conv18")(
        conv17
    )
    conv19 = ConvolutionLayer.ConvolutionLayer(512, (3, 3), (1, 1), 12, name="conv19")(
        conv18
    )
    yolo2_anchors = [[30, 61], [62, 45], [59, 119]]
    yolo2 = YoloLayer.YoloLayer(
        anchors=yolo2_anchors,
        num_classes=1,
        num=3,
        jitter=0.3,
        ignore_threshold=0.7,
        truth_threshold=1.0,
        random=1,
        name="yolo2",
    )(conv19)

    # YOLO LAYER 3
    # conv20
    conv20 = ConvolutionLayer.ConvolutionLayer(128, (1, 1), (1, 1), 12, name="conv20")(
        conv18
    )
    # upsample conv20
    upsample_conv20 = UpSamplingLayer.UpSamplingLayer(factor=2, name="upsample_conv20")(
        conv20
    )
    # route upsample_20 and res3_out
    concat_upconv20_res3out = Concatenate()([upsample_conv20, res3_out])
    conv21 = ConvolutionLayer.ConvolutionLayer(128, (1, 1), (1, 1), 12, name="conv21")(
        concat_upconv20_res3out
    )
    conv22 = ConvolutionLayer.ConvolutionLayer(256, (3, 3), (1, 1), 12, name="conv22")(
        conv21
    )
    conv23 = ConvolutionLayer.ConvolutionLayer(128, (1, 1), (1, 1), 12, name="conv23")(
        conv22
    )
    conv24 = ConvolutionLayer.ConvolutionLayer(256, (3, 3), (1, 1), 12, name="conv24")(
        conv23
    )
    conv25 = ConvolutionLayer.ConvolutionLayer(128, (1, 1), (1, 1), 12, name="conv25")(
        conv24
    )
    conv26 = ConvolutionLayer.ConvolutionLayer(256, (3, 3), (1, 1), 12, name="conv26")(
        conv25
    )
    yolo3_anchors = [[10, 13], [16, 30], [33, 32]]
    yolo3 = YoloLayer.YoloLayer(
        anchors=yolo3_anchors,
        num_classes=1,
        num=3,
        jitter=0.3,
        ignore_threshold=0.7,
        truth_threshold=1.0,
        random=1,
        name="yolov3",
    )(conv26)

    outputs = [yolo1, yolo2, yolo3]
    inputs = [model_input]
    anchors = [yolo1_anchors, yolo2_anchors, yolo3_anchors]

    return Detector(inputs=inputs, outputs=outputs), inputs, outputs, anchors


# class LayerContainer:
#     """LayerContainer."""
#
#     def __init__(self, inputs=None, call=None, outputs=None, is_out=False):
#         """__init__.
#
#         Parameters
#         ----------
#         inputs :
#             inputs
#         call :
#             call
#         outputs :
#             outputs
#         is_out :
#             is_out
#         """
#         self._inputs = inputs
#         self._call_fn = call
#         self._outputs = outputs
#         self._is_out = is_out
#
#     def __call__(self, layer_inputs):
#         """__call__.
#
#         Parameters
#         ----------
#         layer_inputs :
#             layer_inputs
#         """
#         return self._call_fn(layer_inputs)
#
#     @property
#     def inputs(self):
#         """inputs."""
#         return self._inputs
#
#     @inputs.setter
#     def inputs(self, input_args):
#         """inputs.
#
#         Parameters
#         ----------
#         input_args :
#             input_args
#         """
#         self._inputs = input_args
#
#     @property
#     def outputs(self):
#         """outputs."""
#         return self._outputs
#
#     @outputs.setter
#     def outputs(self, output_args):
#         """outputs.
#
#         Parameters
#         ----------
#         output_args :
#             output_args
#         """
#         self._outputs = output_args
#
#     def is_output(self):
#         """is_output."""
#         return self._is_out
#
#     def __repr__(self):
#         return str(self._call_fn.name)
#
#     def __str__(self):
#         return str(self._call_fn.name)
# def build(model_config, input_shape, batch_size):
#    """build.
#        build method processes the model config and other parameters passed as
#        input to the function and returns a detection model. It does the
#        following sequentially:
#            1. iterate through the protobuf network definition and create an
#            integer map that sequentially stores the layer defintion.
#
#    Parameters
#    ----------
#    model_config :
#        model_config
#    input_shape :
#        input_shape
#    batch_size :
#        batch_size
#    """
#    inputs = []
#    outputs = []
#    anchors = []
#
#    # iterate through the network definiton, and create an integer map to
#    # record a sequential order of the layer computation for the model
#    network_def_map = {}
#    network_definition = getattr(model_config, "network_def")
#    for layer_field in network_definition.DESCRIPTOR.fields:
#        layer_field_list = getattr(network_definition, layer_field.name)
#        for layer_protobuf in layer_field_list:
#            network_def_map[int(layer_protobuf.id)] = layer_protobuf
#
#    # iterate through the sorted network definition and parse each individual
#    # layer to generate a keras layer with the provided args through the
#    # protobuf
#    # following function is only used in the local scope to get the input ids
#    # for the layer in the current iteration
#    model_map = OrderedDict()
#
#    def get_inputs(layer_index, layer_protobuf):
#        if layer_protobuf.input_id == -1:
#            if hasattr(layer_protobuf, "indices"):
#                inputs = list(layer_protobuf.indices)
#            else:
#                inputs = layer_index - 1
#        else:
#            inputs = layer_protobuf.input_id
#        return inputs
#
#    # iterating through the network definition and defining model_map
#    for layer_index, layer_def in sorted(network_def_map.items()):
#        layer = layer_builder.parse(layer_def)
#        layer_inputs = get_inputs(layer_index, layer_def)
#        model_map[layer_index] = LayerContainer(
#            inputs=layer_inputs,
#            call=layer,
#            outputs=None,
#            is_out=True if hasattr(layer_def, "anchors") else False,
#        )
#        if hasattr(layer_def, "anchors"):
#            anchors.append(getattr(layer_def, "anchors"))
#
#    def get_layer_inputs(indices):
#        if isinstance(indices, list):
#            return [model_map[index].outputs for index in indices]
#        else:
#            return model_map[indices].outputs
#
#    # call each layer, get its inputs, get and set its outputs w.r.t the model
#    # input shape and batch size provided as argument
#    model_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
#
#    print("Printing model building sequence for debugging")
#    # iterate through the model_map and call.
#    for layer_index, layer in model_map.items():
#        # check if it is the input layer
#        if layer.inputs == 0:  # 0 index represents the input
#            layer_inputs = model_input
#        else:
#            layer_inputs = get_layer_inputs(layer.inputs)
#
#        # compute
#        output = layer(layer_inputs)
#        layer.outputs = output
#        print("layer {}".format(layer))
#        print("layer_input: {}".format(layer_inputs))
#        print("layer_output: {}".format(output))
#
#    # set inputs and outpus to push out a model
#    inputs.append(model_input)
#    outputs = [
#        layer.outputs for layer_index, layer in model_map.items() if layer.is_output()
#    ]
#    anchors = [
#        [
#            [anchor_set[2 * num_anchor], anchor_set[2 * num_anchor + 1]]
#            for num_anchor in range(int(len(anchor_set) / 2))
#        ]
#        for anchor_set in anchors
#    ]
#
#    return Detector(inputs=inputs, outputs=outputs), inputs, outputs, anchors
