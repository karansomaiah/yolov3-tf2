import os
import sys

import numpy as np
import tensorflow as tf
from utils import network_placeholders, config_parser
from models.detector.detector import Detector

if __name__ == "__main__":
    config_file = "/home/knapanda/local/casual/yolov3-tf2/cfg/darknet.cfg"
    parsed_config = config_parser.parse(config_file)
    # for op_i, op in parsed_config:
    #     print(op_i, op)

    sample_image = np.random.randint(0, 255, size=(1, 416, 416, 3), dtype=np.uint8)
    input_image = np.array(sample_image / 255, dtype=np.float32)
    input_tensor = tf.convert_to_tensor(input_image)

    detection_model = Detector(parsed_config, 1)
    detection_model = detection_model.build_model(input_image.shape[1:])

    detection_model.summary()
    detection_out = detection_model(input_tensor)
    for out_index, out in detection_out:
        with open(
            os.path.join(
                "/home/knapanda/local/object_detection/checkpoints/out_{}.npy".format(
                    out_index
                )
            ),
            "wb",
        ) as f:
            np.save(f, out)

    # save model
    saved_model_dir = (
        "/home/knapanda/local/object_detection/checkpoints/saved_model/my_model/"
    )
    detection_model.save(saved_model_dir)

    # convert model to trt
    # trt_model_dir = "/home/knapanda/local/object_detection/checkpoints/trt/"
    # params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP16")
    # converter = tf.experimental.tensorrt.Converter(
    #     input_saved_model_dir=saved_model_dir, conversion_params=params
    # )
    # converter.convert()
    # converter.save(trt_model_dir)
