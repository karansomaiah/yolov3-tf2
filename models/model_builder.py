from collections import OrderedDict, namedtuple

# tensorflow imports
import tensorflow as tf
import tensorflow.keras as keras

# protobuf imports
from protos import (
    convolution_pb2,
    residual_pb2,
    route_pb2,
    upsample_pb2,
    yolo_pb2,
    network_pb2,
)
import layer_builder  # module containing building functionality for layers


class LayerContainer:
    def __init__(self, inputs=None, call=None, outputs=None, is_out=False):
        self._inputs = inputs
        self._call_fn = call
        self._outputs = outputs
        self._is_out = is_out

    def __call__(self, layer_inputs):
        return self._call_fn(layer_inputs)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, input_args):
        self._inputs = input_args

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, output_args):
        self._outputs = output_args

    def is_output(self):
        return self._is_out


class DetectorModel(keras.Model):
    """Detector Model."""

    def __init__(self, model_config):  # ... arguments are pending
        self.model_config_proto = model_config
        self.model_config_map = {}
        self.layer_def = OrderedDict()
        self.model_outputs = []
        self.parse_model_config()
        self.anchors_list = []

    def call(self, inputs):
        for layer_index, layer in self.layer_def.items():
            layer_inputs = self.get_inputs_for_layer(layer.inputs)
            layer_output = layer(layer_inputs)
            layer.outputs = layer_output
        return self.model_outputs

    def build_model(self, input_shape, batch_size=None):
        model_input = keras.layers.Input(shape=input_shape, batch_size=batch_size)
        for layer_index, layer_cfg in sorted(self.model_config_map.items()):
            layer = layer_builder.parse(layer_cfg)
            inputs = []
            if layer_cfg.input_id == -1:
                if hasattr(layer_cfg, "indices"):
                    inputs = list(layer_cfg.indices)
                else:
                    inputs = layer_index - 1
            else:
                inputs = layer_cfg.input_id
            self.layer_def[layer_index] = LayerContainer(
                inputs=inputs,
                call=layer,
                outputs=None,
                is_out=True if hasattr(layer_cfg, "anchors") else False,
            )
            if hasattr(layer_cfg, "anchors"):
                self.anchors_list.append(self.layer_def[layer_index].anchors)
        return keras.Model(inputs=[model_input], outputs=self.call(model_input))

    def parse_model_config(self):
        for pb_field in self.model_config_proto.network_def.DESCRIPTOR.fields:
            layer_cfg_list = getattr(
                self.model_config_proto.network_def, pb_field.name
            )  # get list of particular layer class
            for layer_cfg in layer_cfg_list:
                self.model_config_map[int(layer_cfg.id)] = layer_cfg
        return

    def get_inputs_for_layer(self, indices):
        if isinstance(indices, list):
            layer_output_as_input = [self.layer_def[index].outputs for index in indices]
        else:
            layer_output_as_input = self.layer_def[indices].outputs
        return layer_output_as_input
