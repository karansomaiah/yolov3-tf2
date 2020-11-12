#!/usr/bin/env python
import cv2

# CONSTANTS
LAYER_TYPES = ["convolution", "residual", "connection", "route", "yolo", "upsample"]

# UTILITY FUNCTIONS
def read_image(file_path, size):
    image = cv2.imread(file_path)
    if image.shape[:-1] != size:
        image = cv2.resize(image, size)