import os
import sys

import tensorflow as tf

# random import inits
from utils import clean_imports, proto_utils
from models.model_builder import DetectorModel
import train
import infer

clean_imports.clean()  # remove ros opencv
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


FLAGS = tf.compat.v1.flags.FLAGS
# Flags are defined in the format 'attribute','default_value', """Description"""
tf.compat.v1.flags.DEFINE_string(
    "config_filepath",
    "./configs/darknet53.config",
    """Path to config file to kick off training""",
)
tf.compat.v1.flags.DEFINE_bool(
    "train", True, """Boolean specifying whether to train or not"""
)
tf.compat.v1.flags.DEFINE_bool(
    "evaluate", True, """Boolean specifying whether to evaluate or not"""
)
tf.compat.v1.flags.DEFINE_bool(
    "predict", True, """Boolean specifying whether to predict or not"""
)
tf.compat.v1.flags.DEFINE_string(
    "checkpoint_path",
    "",
    """Path specifying the checkpoint directory
                                 for evaluation. The directory should have at
                                 least 1 checkpoint. This path is useful when
                                 only evaluating on a list of checkpoints and
                                 not training. Given training is true,
                                 evaluation will happen at each epoch of
                                 training.""",
)


def scheduler(config_protobuf, is_training, is_evaluating, if_predicting):
    # start scheduling any of the following
    # 1. training only
    # 2. training + evalutation
    # 3. evalutaion only (given a bunch of checkpoints, just evaluate on a
    # dataset)
    # 4. infer
    if is_training or is_evaluating:
        train.trainer()


def main(argv):
    print("Loading flags..")
    print("config path flag: {}".format(FLAGS.config_filepath))

    # get arguments
    config_filepath = FLAGS.config_filepath
    is_training = bool(FLAGS.train)
    is_evaluating = bool(FLAGS.train)
    is_predicting = bool(FLAGS.predict)

    # read the config file
    config_pb = proto_utils.read_protobuf_config(config_filepath)
    scheduler(config_pb, is_training, is_evaluating, is_predicting)


# main loop
if __name__ == "__main__":
    tf.compat.v1.app.run()