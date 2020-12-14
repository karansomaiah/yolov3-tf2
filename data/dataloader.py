#!/usr/bin/env python
"""
Dataloader functionality for datasets
"""

import os
import json
import numpy as np
import tensorflow as tf

import PIL
from PIL import Image

# import data.tfrecord_creator as tfrecord_creator

class_map = {"Dog": 0}


def read_label_json(filepath):
    """
    Will contain
    """
    with open(filepath, "r") as filereader:
        labelled_data = json.load(filereader)
    return labelled_data


def annotation_generator(annotations, image_width, image_height):
    for annotation in annotations:
        image = Image.open(annotation["image_path"])  # read image
        image = image.resize(
            size=(image_width, image_height), resample=PIL.Image.LANCZOS
        )
        bboxes = np.array(
            [
                [
                    bbox[0] / image_width,
                    bbox[1] / image_height,
                    bbox[2] / image_width,
                    bbox[3] / image_height,
                ]
                # [bbox[0]/640, bbox[1]/640, bbox[2]/640, bbox[3]/640]
                for bbox in annotation["bboxes"]
            ]
        )
        classes = np.array(
            [[class_map[class_label]] for class_label in annotation["classes"]]
        )
        label = np.concatenate([bboxes, classes], axis=-1)
        yield image, label


def combine(images, labels):
    images = tf.stack(images, axis=0) / 255.0
    labels_list_with_batch = []
    for batch_index, label in enumerate(labels):
        labels_list_with_batch.append(
            tf.concat(
                [tf.expand_dims(tf.ones(label.shape[0]) * batch_index, axis=-1), label],
                axis=-1,
            )
        )
    return images, tf.concat(labels_list_with_batch, axis=0)


def tf_dataloader(config_filepath):
    # Get Values from the config
    input_height = config_filepath["INPUT_H"]
    input_width = config_filepath["INPUT_W"]
    annotation_filename = config_filepath["annotation_path"]
    annotations = read_label_json(annotation_filename)
    # create_tfrecords = bool(int(config_filepath["create_tfrecords"]))

    #    ---------------------------------------------------------------------------
    #    # create tfrecords
    #    if create_tfrecords:
    #        annotations = read_label_json(annotation_filename)
    #        tfrecord_creator.write_tfrecords("/home/knapanda/",
    #            annotations, input_height, input_width
    #        )
    #
    #    tfrecord_features = {
    #        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    #        'x': tf.io.VarLenFeature(tf.float32),
    #        'y': tf.io.VarLenFeature(tf.float32),
    #        'w': tf.io.VarLenFeature(tf.float32),
    #        'h': tf.io.VarLenFeature(tf.float32),
    #        'label': tf.io.VarLenFeature(tf.int64),
    #    }
    #
    #    def _convert(parsed_example):
    #        return{
    #            'image': tf.reshape(
    #                tf.io.decode_raw(parsed_example['image'], tf.uint8),
    #                [416, 416, 3]),
    #            'x': tf.sparse.to_dense(parsed_example['x']),
    #            'y': tf.sparse.to_dense(parsed_example['y']),
    #            'w': tf.sparse.to_dense(parsed_example['w']),
    #            'h': tf.sparse.to_dense(parsed_example['h'])
    #        }
    #
    #    def _parse_example(example):
    #        return _convert(tf.io.parse_single_example(example, tfrecord_features))
    #    ---------------------------------------------------------------------------

    #    ---------------------------------------------------------------------------
    #    # processed labels
    #    train_labels = process_labels(train_labels)
    #    val_labels = process_labels(val_labels)
    #
    #    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    #    tf_train_dataset = tf_train_dataset.map(reader)
    #    tf_train_dataset = tf_train_dataset.repeat(90)
    #    tf_train_dataset = tf_train_dataset.batch(5)
    #
    #    tf_val_dataset = tf.data.Dataset.from_tensor_slices((val_images,
    #                                                         val_labels))
    #    tf_val_dataset = tf_val_dataset.map(reader)
    #    tf_val_dataset = tf_val_dataset.repeat(10)
    #    tf_val_dataset = tf_val_dataset.batch(5)

    #    return tf_train_dataset, tf_val_dataset

    #    ---------------------------------------------------------------------------
    train_dataset = tf.data.Dataset.from_generator(
        lambda: annotation_generator(
            annotations["annotations"]["train"], input_width, input_height
        ),
        (tf.float32, tf.float32),
    )
    train_dataset = train_dataset.repeat(10)
    validation_dataset = tf.data.Dataset.from_generator(
        lambda: annotation_generator(
            annotations["annotations"]["validation"], input_width, input_height
        ),
        (tf.float32, tf.float32),
    )
    validation_dataset = validation_dataset.repeat(10)

    return train_dataset, validation_dataset
