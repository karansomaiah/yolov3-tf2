#!/usr/bin/env python
"""
Dataloader functionality for datasets
"""

import json
import tensorflow


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

    image_filenames = []
    image_annotations = []

    for annotation_object in annotations:
        image_filenames.append(annotation_object["image_path"])
        image_annotations.append(
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

    return image_filenames, image_annotations


def load_annotations(label_filepath, input_height, input_width):
    """"""
    annotation = read_label_json(label_filepath)
    images, labels = process_annotation(annotation, input_height, input_width)
    return images, labels


def tf_dataloader(config_filepath):
    input_height = config_filepath["INPUT_H"]
    input_width = config_filepath["INPUT_W"]
    annotation_filename = config_filepath["annotation_path"]
    images, labels = load_annotations(annotation_filename, input_height, input_width)

    # need to create images, labels structure either here or up above...
