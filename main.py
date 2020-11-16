import os
import sys

import numpy as np

from utils import network_placeholders, config_parser
from models.detector.detector import Detector

if __name__ == "__main__":
    config_file = "/home/knapanda/local/casual/yolov3-tf2/cfg/darknet.cfg"
    parsed_config = config_parser.parse(config_file)

    detection_model = Detector(parsed_config, 1)

    sample_image = np.random.randint(0, 255, size=(1, 416, 416, 3), dtype=np.uint8)
    input_image = np.array(sample_image / 255, dtype=np.float32)

    detection_out = detection_model(input_image)
    print(detection_out)