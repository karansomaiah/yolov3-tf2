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
import models.layer_builder as layer_builder  # module containing building functionality for layers
from models.loss.detection_loss import DetectionLoss


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

    def __init__(self, model_config, num_classes):  # ... arguments are pending
        super(DetectorModel, self).__init__()
        self.model_config_proto = model_config
        self.num_classes = num_classes
        self.model_config_map = {}
        self.layer_def = OrderedDict()
        self.model_outputs = []
        self.parse_model_config()
        self.anchors_list = []
        self.compiled_loss = None

    def call(self, inputs):
        for layer_index, layer in self.layer_def.items():
            if not layer.inputs == 0:
                layer_inputs = self.get_inputs_for_layer(layer.inputs)
                layer_output = layer(layer_inputs)
            else:
                layer_inputs = inputs
                layer_output = layer(layer_inputs)
            layer.outputs = layer_output
            if layer.is_output():
                self.model_outputs.append(layer_output)
        return self.model_outputs

    def build_model(self, input_shape, batch_size=None):
        image_height, image_width, num_channels = input_shape
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
                self.anchors_list.append(layer_cfg.anchors)

        # self.compiled_loss = DetectionLoss(
        #    image_height, image_width, self.num_classes, self.anchors_list
        # )

        return (
            keras.Model(inputs=[model_input], outputs=self.call(model_input)),
            self.anchors_list,
        )

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

    # def compile(self, optimizer, loss):
    #    super(DetectorModel, self).compile()
    #    self.optimizier = optimizer
    #    self.compiled_loss = loss

    # single train step
    def train_step(self, data):
        images, labels = data
        train_batch_size = images.shape[0]
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = 0
            for anchor_index, anchor in enumerate(self.anchors_list):
                current_loss = self.compiled_loss(
                    labels, predictions[anchor_index], anchor_index
                )
                loss += current_loss
            # loss = self.compiled_loss(labels, predictions)
            # all_losses = self.compiled_loss.get_all_losses()
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return {"loss": loss}

    # single test step
    def eval_step(self, data):
        images, labels = data
        eval_batch_size = images.shape[0]
        predictions = self(images, training=False)
        loss = 0
        for anchor_index, anchor in enumerate(self.anchors_list):
            current_loss = self.compiled_loss(
                labels, predictions[anchor_index], anchor_index
            )
            loss += current_loss
        # loss = self.compiled_loss(labels, predictions)
        return {"loss": loss}

    @property
    def metrics(self):
        return [self.compiled_loss]

    def postprocess_detections(self, detections):
        detections_reshaped = []
        for detection_index, detection in enumerate(detections):
            batch, g_x, g_y, channels = tf.shape(detection)
            channels_wo_anchors = int(
                channels / len(self.anchors_list[detection_index])
            )
            detection_reshaped = tf.reshape(
                detection,
                [
                    batch,
                    len(self.anchors_list[detection_index]),
                    g_x,
                    g_y,
                    channels_wo_anchors,
                ],
            )
            detections_reshaped.append(detection_reshaped)

        detection_boxes = tf.concat(
            [
                tf.reshape(detection[:, :, :, :, :4], [-1, 4])
                for detection in detections
            ],
            axis=0,
        )
        detection_scores = tf.concat(
            [
                tf.reshape(
                    detection[:, :, :, :, 4],
                    [
                        -1,
                    ],
                )
                for detection in detections
            ],
            axis=0,
        )
        nms_indices = tf.image.non_max_suppression(
            boxes=detection_boxes, scores=detection_scores, max_output_size=100
        )
        return tf.gather(detection_boxes, nms_indices)
