#!/usr/bin/env python
"""
Dataloader functionality for datasets
"""

import json
import tensorflow

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class_map = {"Dog": 1}


def read_label_json(filepath):
    """
    Will contain
    """
    with open(filepath, "r") as filereader:
        labelled_data = json.load(filereader)
    return labelled_data


def process_annotation(annotation_dict, input_height, input_width):
    """
    Processes the dictionary containing annotation information
    and returns classes dictionary, as well as the annotation
    """
    class_map = annotation_dict["classes"]
    annotations = annotation_dict["annotations"]

    train_image_filenames = []
    train_image_annotations = []
    val_image_filenames = []
    val_image_annotations = []

    for annotation_object in annotations["train"]:
        train_image_filenames.append(annotation_object["image_path"])
        train_image_annotations.append(
            {
                "bboxes": [
                    [
                        box[0] / input_height,
                        box[1] / input_width,
                        box[2] / input_height,
                        box[3] / input_width,
                    ]
                    for box in annotation_object["bboxes"]
                ],
                "classes": [
                    class_map[class_label]
                    for class_label in annotation_object["classes"]
                ],
            }
        )

    for annotation_object in annotations["validation"]:
        val_image_filenames.append(annotation_object["image_path"])
        val_image_annotations.append(
            {
                "bboxes": [
                    [
                        box[0] / input_height,
                        box[1] / input_width,
                        box[2] / input_height,
                        box[3] / input_width,
                    ]
                    for box in annotation_object["bboxes"]
                ],
                "classes": [
                    class_map[class_label]
                    for class_label in annotation_object["classes"]
                ],
            }
        )

    return (
        train_image_filenames,
        train_image_annotations,
        val_image_filenames,
        val_image_annotations,
    )


def process_labels(labels):
    labels_numpy = []
    for label in labels:
        num = len(label["classes"])
        label_bbox_np = label["bboxes"]
        label_class_np = np.expand_dims(
            np.array(label["classes"], dtype=np.float32), axis=1
        )
        label_np = np.concatenate((label_bbox_np, label_class_np), axis=0)
        labels_numpy.append(label_np)
    return label_np


def tf_dataloader(config_filepath):
    # Get Values from the config
    input_height = config_filepath["INPUT_H"]
    input_width = config_filepath["INPUT_W"]
    annotation_filename = config_filepath["annotation_path"]

    # read annotations
    annotations = read_label_json(annotation_filename)
    train_images, train_labels, val_images, val_labels = process_annotation(
        annotation_filename, input_height, input_width
    )

    # processed labels
    labels = process_labels(labels)

    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    tf_val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
