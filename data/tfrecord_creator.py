#!/usr/bin/env python
import json
import numpy as np
import tensorflow as tf

import PIL
from PIL import Image

from utils import proto_utils
import dataloader_v2 as dataloader

# defining input argument flags
FLAGS = tf.compat.v1.flags.FLAGS

# args for config file which contains class map
tf.compat.v1.flags.DEFINE_string(
    "config_filepath",
    "./configs/darknet53.config",
    """Path to config file"""
    """ which consists of the training and validation config""",
)
# args for providing the path to the annotation file
tf.compat.v1.flags.DEFINE_string(
    "annotation_file",
    "",
    """Path to the annotation file""",
)
# args for providing the path to output the record file,
# else saves in the current path
tf.compat.v1.flags.DEFINE_string(
    "output_path",
    "dataset.record",
    """Path to write the TensorflowRecord file to.""",
)


def _bytes_feature(value):
    """_bytes_feature.
    Returns a bytes_list from a string / byte.

    :param value: value to be converted to a bytes feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature_list(value):
    """_bytes_feature_list.
    Returns a bytes_list from a string / byte.

    :param value: list of values to be converted to a bytes feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """_float_feature.
    Returns a float_list from a float / double.

    :param value: value to be converted to a float feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(value):
    """_float_feature_list.
    Returns a float_list from a float / double.

    :param value: list of values to be converted to a float feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """_int64_feature.
    Returns an int64_list from a bool / enum / int / uint.

    :param value: value to be converted into a int64 feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(value):
    """_int64_feature_list.
    Returns an int64_list from a bool / enum / int / uint.

    :param value: list of values to be converted into a int64 feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_serialized_example(filename, annotation, input_width, input_height):
    """get_serialized_example.
    Serializes each annotation into a Tensorflow example.
        i.e. tf.train.Example()

    :param filename: string describing the filename
    :param annotation: a dictionary with keys storing the annotation 
                       for the current image.
                       example:
                        {
                            'bb': [[0, 0, 0, 0],
                                   [10, 20, 30, 40]],
                            'classes': ['Dog', 'Cat']
                        }
    :param input_width: the input image width for training.
    :param input_height: the input image height for training.
    """
    # image stuff
    image = Image.open(filename)
    image = image.resize(size=(input_width, input_height), resample=PIL.Image.LANCZOS)
    image_string = np.array(image).tostring()

    # bounding boxes
    x_list, y_list, w_list, h_list = [], [], [], []
    labels = []

    num_labels = len(annotation["classes"])
    for label_index in range(num_labels):
        bbox = annotation["bboxes"][label_index]
        class_label = int(annotation["classes"][label_index])
        x, y, w, h = bbox
        x_list.append(x)
        y_list.append(y)
        w_list.append(w)
        h_list.append(h)
        labels.append(class_label)

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": _bytes_feature(image_string),
                "x": _float_feature_list(x_list),
                "y": _float_feature_list(y_list),
                "w": _float_feature_list(w_list),
                "h": _float_feature_list(h_list),
                "label": _int64_feature_list(labels),
            }
        )
    )


def write_tfrecords(record_path, annotation_dict, class_map, input_width, input_height):
    """write_tfrecords.
    Function to write Tensorflow records given a file path, a list of annotations,
    a dictionary mapping classes to integers, and the image dimensions.

    :param record_path: string describing the output file path.
    :param annotation_dict: dictionary holding the list of annotations 
                            to be written to a Tensorflow Record File.
    :param class_map: dictionary storing mapping from string classes to integers.
    :param input_width: input image width.
    :param input_height: input image height. 
    """
    """
    Processes the dictionary containing annotation information
    and returns classes dictionary, as well as the annotation
    """
    with tf.io.TFRecordWriter(record_path) as record_writer:
        for annotation_object in annotation_dict:
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
            example = get_serialized_example(
                current_filename, current_annotation, input_width, input_height
            )
            record_writer.write(example.SerializeToString())
        print("Completed Writing TFRecord {}".format(record_path))
    return


def main(argv):
    """main.
    Main function running writing to tensorflow records.

    :param argv: arguments provided for training
    """
    config_filepath = FLAGS.config_filepath
    annotation_path = FLAGS.annotation_file
    output_path = FLAGS.output_path

    # read the config and generate the class map
    config = proto_utils.read_protobuf_config(config_filepath)
    class_map = dataloader.generate_class_map(config.class_def)

    # read the annotation file
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    write_tfrecords(
        output_path,
        annotations,
        class_map,
        config.model_specs.image_width,
        config.model_specs.image_height,
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()
