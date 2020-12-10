#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf

import PIL
from PIL import Image


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature_list(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_serialized_example(filename, annotation, input_width, input_height):
    # image stuff
    image = Image.open(filename)
    image = image.resize(size=(input_width, input_height),
                         resample=PIL.Image.LANCZOS)
    image_string = np.array(image).tostring()

    # bounding boxes
    x_list, y_list, w_list, h_list = [], [], [], []
    labels = []

    num_labels = len(annotation['classes'])
    for label_index in range(num_labels):
        bbox = annotation['bboxes'][label_index]
        class_label = int(annotation['classes'][label_index])
        x, y, w, h = bbox
        x_list.append(x)
        y_list.append(y)
        w_list.append(w)
        h_list.append(h)
        labels.append(class_label)


    return tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image_string),
        'x': _float_feature_list(x_list),
        'y': _float_feature_list(y_list),
        'w': _float_feature_list(w_list),
        'h': _float_feature_list(h_list),
        'label': _int64_feature_list(labels)
    }))


def write_tfrecords(directory, annotation_dict, input_width, input_height):
    """
    Processes the dictionary containing annotation information
    and returns classes dictionary, as well as the annotation
    """
    class_map = annotation_dict["classes"]
    annotations = annotation_dict["annotations"]

    if 'train' in annotations.keys():
        train_filename = os.path.join(directory, "train.record")
        with tf.io.TFRecordWriter(train_filename) as record_writer:        
            for annotation_object in annotations["train"]:
                current_filename = annotation_object["image_path"]
                current_annotation = {
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
                example = get_serialized_example(current_filename,
                                                 current_annotation,
                                                 input_width,
                                                 input_height)
                record_writer.write(example.SerializeToString())
        print("Completed Writing TFRecord {}".format(train_filename))

    if 'validation' in annotations.keys():
        validation_filename = os.path.join(directory, "validation.record")
        with tf.io.TFRecordWriter(validation_filename) as writer:
            for annotation_object in annotations["validation"]:
                current_filename = annotation_object["image_path"]
                current_annotation = {
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
                example = get_serialized_example(current_filename,
                                                 current_annotation,
                                                 input_width,
                                                 input_height)
                writer.write(example.SerializeToString())
        print("Completed Writing TFRecord {}".format(validation_filename))
    print("Completed Creating TFRecords")

