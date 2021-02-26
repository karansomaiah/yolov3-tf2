from setuptools import setup, find_packages

setup(
    name="yolov3-tf2",
    version="0.1.0",
    description="Setting up yolov3 code written in tensorflow",
    packages=find_packages(include=["data", "models", "protos", "utils"]),
)
