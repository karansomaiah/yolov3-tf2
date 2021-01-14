import tensorflow as tf
from google.protobuf import text_format
from protos import convolution_pb2, network_pb2, residual_pb2, route_pb2, upsample_pb2, yolo_pb2


def read_protobuf_config(config_filepath):
    config_pb = network_pb2.NetworkDefinition()
    with tf.io.gfile.GFile(config_filepath, 'r') as fid:
        config_string = fid.read()
        text_format.Merge(config_string, config_pb)
    return config_pb
