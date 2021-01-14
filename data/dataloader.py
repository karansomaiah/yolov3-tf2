#!/usr/bin/env python
"""
Dataloader functionality for datasets
"""

import os
import json
import random
import numpy as np
import tensorflow as tf

import PIL
from PIL import Image


def generate_class_map(config):
    return {
        class_proto.class_name: class_proto.clas_id for class_proto in config.class_map
    }


def read_label_json(filepath):
    """
    Will contain
    """
    with open(filepath, "r") as filereader:
        labelled_data = json.load(filereader)
    return labelled_data


def annotation_generator(
    annotations, class_map, image_width, image_height, batch_size, shuffle=False
):
    # apply shuffle to the dataset when asked for
    random.shuffle(annotations)

    # generate batch indices given a batch_size
    annotation_length = len(annotations)
    for annotation_ndx in range(0, annotation_length, batch_size):
        batched_images_as_list = []
        batched_labels_as_list = []
        for batch_index, batch_num in enumerate(
            range(annotation_ndx, min(annotation_ndx + batch_size, annotation_length))
        ):
            image = Image.open(annotations[batch_num]["image_path"])
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
                    for bbox in annotations[batch_num]["bboxes"]
                ]
            )
            classes = np.array(
                [
                    [class_map[class_label]]
                    for class_label in annotations[batch_num]["classes"]
                ]
            )
            label = np.concatenate([bboxes, classes], axis=-1)
            batched_labels_as_list.append(
                np.concatenate(
                    [
                        np.expand_dims(np.ones(label.shape[0]) * batch_index, axis=-1),
                        label,
                    ],
                    axis=-1,
                )
            )
            batched_images_as_list.append(image)

        # combine them
        batch_images = np.stack(batched_images_as_list, axis=0)
        batch_labels = np.concatenate(batched_labels_as_list, axis=0)
        yield batch_images, batch_labels


"""
DEPRACATED
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
"""

""" 
@DEPRACATED
def tf_dataloader(config_filepath):
    # Get Values from the config
    input_height = config_filepath["INPUT_H"]
    input_width = config_filepath["INPUT_W"]
    annotation_filename = config_filepath["annotation_path"]
    annotations = read_label_json(annotation_filename)
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
"""


def tf_dataloader_v2(
    label_file_path, class_map, image_height, image_width, batch_size, shuffle=True
):
    # get class_map
    if label_file_path:
        annotations = read_label_json(label_file_path)
        train_dataset = tf.data.Dataset.from_generator(
            lambda: annotation_generator(
                annotations,
                class_map,
                input_width,
                input_height,
            ),
            (tf.float32, tf.float32),
        )
        return train_dataset
    else:
        return None
