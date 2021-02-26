## Implementation for YOLOV3 written in Tensorflow 2.

It's currently a work in progress!

* To compile protobuf:
> protoc --python_out=. protos/*.proto

* To create a dataset:
> python data/tfrecord_creator.py --config_filepath=configs/darknet53.config --annotation_file=data/sample/validation.json --output_path=data/sample/validation.record

* To run training:
> python model_main.py --config_filepath=configs/darknet53.config --train --evaluate
