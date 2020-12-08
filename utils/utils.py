#!/usr/bin/env python
"""
"""
import cv2
import tensorflow as tf
from collections import namedtuple

# CONSTANTS
INPUT_H = 416
INPUT_W = 416
NUM_CHANNELS = 3
IGNORE_LAYER_INDEX = -1
DEFAULT_LAYER_ARG = -1
LAYER_TYPES = [
    "convolution", "residual", "connection", "route", "yolo", "upsample"
]

# DataTypes
LayerTuple = namedtuple("LayerTuple", ["call", "args", "index"])
INT = tf.int32
FLOAT = tf.float32
BOOL = tf.bool


# UTILITY FUNCTIONS
def read_image(file_path, size):
    image = cv2.imread(file_path)
    if image.shape[:-1] != size:
        image = cv2.resize(image, size)
    return image
