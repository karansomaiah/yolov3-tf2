#!/usr/bin/env python
"""
"""

# regular imports
from collections import OrderedDict

# tensorflow imports
import tensorflow as tf
import tensorflow.keras as tf_keras

# protobufs
# from protos import (
#    convolution_pb2,
#    residual_pb2,
#    route_pb2,
#    upsample_pb2,
#    yolo_pb2,
#    network_pb2,
# )

import models.layer_builder as layer_builder
from models.loss.detection_loss import DetectionLoss


class LayerContainer:
    """LayerContainer."""

    def __init__(self, inputs=None, call=None, outputs=None, is_out=False):
        """__init__.

        Parameters
        ----------
        inputs :
            inputs
        call :
            call
        outputs :
            outputs
        is_out :
            is_out
        """
        self._inputs = inputs
        self._call_fn = call
        self._outputs = outputs
        self._is_out = is_out

    def __call__(self, layer_inputs):
        """__call__.

        Parameters
        ----------
        layer_inputs :
            layer_inputs
        """
        return self._call_fn(layer_inputs)

    @property
    def inputs(self):
        """inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, input_args):
        """inputs.

        Parameters
        ----------
        input_args :
            input_args
        """
        self._inputs = input_args

    @property
    def outputs(self):
        """outputs."""
        return self._outputs

    @outputs.setter
    def outputs(self, output_args):
        """outputs.

        Parameters
        ----------
        output_args :
            output_args
        """
        self._outputs = output_args

    def is_output(self):
        """is_output."""
        return self._is_out


class Detector(tf_keras.Model):
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
            loss2 = self.loss_fn1(labels, output2)
            loss3 = self.loss_fn1(labels, output3)
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
        loss2 = self.loss_fn1(labels, output2)
        loss3 = self.loss_fn1(labels, output3)
        loss = loss1 + loss2 + loss3
        return {"loss": loss}


def build(model_config, input_shape, batch_size):
    """build.
        build method processes the model config and other parameters passed as
        input to the function and returns a detection model. It does the
        following sequentially:
            1. iterate through the protobuf network definition and create an
            integer map that sequentially stores the layer defintion.

    Parameters
    ----------
    model_config :
        model_config
    input_shape :
        input_shape
    batch_size :
        batch_size
    """
    inputs = []
    outputs = []
    anchors = []

    # iterate through the network definiton, and create an integer map to
    # record a sequential order of the layer computation for the model
    network_def_map = {}
    network_definition = getattr(model_config, "network_def")
    for layer_field in network_definition.DESCRIPTOR.fields:
        layer_field_list = getattr(network_definition, layer_field.name)
        for layer_protobuf in layer_field_list:
            network_def_map[int(layer_protobuf.id)] = layer_protobuf

    # iterate through the sorted network definition and parse each individual
    # layer to generate a keras layer with the provided args through the
    # protobuf
    # following function is only used in the local scope to get the input ids
    # for the layer in the current iteration
    model_map = OrderedDict()

    def get_inputs(layer_index, layer_protobuf):
        if layer_protobuf.input_id == -1:
            if hasattr(layer_protobuf, "indices"):
                inputs = list(layer_protobuf.indices)
            else:
                inputs = layer_index - 1
        else:
            inputs = layer_protobuf.input_id
        return inputs

    # iterating through the network definition and defining model_map
    for layer_index, layer_def in sorted(network_def_map.items()):
        layer = layer_builder.parse(layer_def)
        layer_inputs = get_inputs(layer_index, layer_def)
        model_map[layer_index] = LayerContainer(
            inputs=layer_inputs,
            call=layer,
            outputs=None,
            is_out=True if hasattr(layer_def, "anchors") else False,
        )
        if hasattr(layer_def, "anchors"):
            anchors.append(getattr(layer_def, "anchors"))

    def get_layer_inputs(indices):
        if isinstance(indices, list):
            return [model_map[index].outputs for index in indices]
        else:
            return model_map[indices].outputs

    # call each layer, get its inputs, get and set its outputs w.r.t the model
    # input shape and batch size provided as argument
    model_input = tf_keras.layers.Input(shape=input_shape, batch_size=batch_size)
    # iterate through the model_map and call.
    for layer_index, layer in model_map.items():
        # check if it is the input layer
        if layer.inputs == 0:  # 0 index represents the input
            layer_inputs = model_input
        else:
            layer_inputs = get_layer_inputs(layer.inputs)

        # compute
        output = layer(layer_inputs)
        layer.outputs = output

    # set inputs and outpus to push out a model
    inputs.append(model_input)
    outputs = [
        layer.outputs for layer_index, layer in model_map.items() if layer.is_output()
    ]
    anchors = [
        [
            [anchor_set[2 * num_anchor], anchor_set[2 * num_anchor + 1]]
            for num_anchor in range(int(len(anchor_set) / 2))
        ]
        for anchor_set in anchors
    ]

    return Detector(inputs=inputs, outputs=outputs), inputs, outputs, anchors
