import os
import sys
import numpy as np

CHECK_OPENCV2 = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if CHECK_OPENCV2 in sys.path:
    sys.path.remove(CHECK_OPENCV2)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from utils import network_placeholders, config_parser
from models.detector.detector import Detector

import onnx
import keras2onnx

## save model
# saved_model_dir = "/home/karan/Checkpoint/ckpt-45"
#
#
## convert model to trt
# trt_model_dir = "/home/karan/Checkpoint/TRT"
# params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP16")
# converter = tf.experimental.tensorrt.Converter(
#    input_saved_model_dir=saved_model_dir, conversion_params=params
# )
# converter.convert()
# converter.save(trt_model_dir)


# trained_checkpoint_prefix = "/home/karan/Checkpoint/ckpt-45"
# export_directory = "/home/karan/Checkpoint/SavedModel"
#
#
# graph = tf.Graph()
# with tf.compat.v1.Session(graph=graph) as sess:
#    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix +
#                                                  '.meta')
#    loader.restore(sess, trained_checkpoint_prefix)
#
#    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_directory)
#    builder.add_meta_graph_and_variables(sess,
#                                        [tf.saved_model.TRAINING,
#                                         tf.saved_model.SERVING],
#                                        strip_default_attrs=True)
#    builder.save()

model_config_file = "cfg/darknet.cfg"
parsed_config = config_parser.parse(model_config_file)

detection_model = Detector(parsed_config, 1)
detection_model = detection_model.build_model((416, 416, 3), 1)
detection_model.layers[0].trainable = False

checkpoint = tf.train.Checkpoint(model=detection_model)
latest_checkpoint_path = tf.train.latest_checkpoint("/home/karan/Checkpoint/")
checkpoint.restore(latest_checkpoint_path)

# shape = (1, 416, 416, 3)
# sample_image = np.random.randint(0, 255, size=(1, 416, 416, 3), dtype=np.uint8)
# input_image = np.array(sample_image / 255, dtype=np.float32)

# model = tf.keras.models.load_model("/home/karan/Checkpoint/")
# model.build_model(shape[1:], shape[0])

print(detection_model.input)
print(detection_model.layers[0])
onnx_model = keras2onnx.convert_keras(detection_model, detection_model.name)


file = open("/home/karan/Checkpoint/TRT/model.onnx", "wb")
file.write(onnx_model.SerializeToString())
file.close()


# print("Completed Building and saved SaveDModel to {}".format(export_directory))
