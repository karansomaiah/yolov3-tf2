import os
import sys

from utils import clean_imports

clean_imports.clean()  # clean ROS OpenCV import and disable tensorflow logs


CHECK_OPENCV2 = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if CHECK_OPENCV2 in sys.path:
    sys.path.remove(CHECK_OPENCV2)

# disable logging for TF Debugging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import numpy as np
import tensorflow as tf
from utils import network_placeholders, config_parser
from models.detector.detector import Detector
from models.loss import detection_loss as loss_fn
from data import dataloader
from cfg import config_reader


# Constants
GLOBAL_BATCH_SIZE = 4
NUM_GPUS = 2
BATCH_SIZE = int(GLOBAL_BATCH_SIZE / NUM_GPUS)
NUM_EPOCHS = 50
NUM_REPLICAS = 1


if __name__ == "__main__":
    model_config_file = "cfg/darknet.cfg"
    parsed_config = config_parser.parse(model_config_file)
    training_config = config_reader.reader("cfg/config.json")
    train_dataset, validation_dataset = dataloader.tf_dataloader(training_config)

    # ------------------ START OF DISTRIBUTED TRAINING -------------------------#
    # training
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        buffer_size = 1000

        batch_counter = 0
        batch_images = []
        batch_labels = []

        detection_model = Detector(parsed_config, 1)
        detection_model = detection_model.build_model((416, 416, 3))
        # detection_model.summary()
        detection_model.layers[0].trainable = False
        # for layer in detection_model.layers:
        #    print(layer.name, layer.trainable)

        optimizer = tf.keras.optimizers.Adam()
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9)
        saver = tf.train.Checkpoint(optimizer=optimizer, model=detection_model)

        def train_step(inputs):
            images, labels = inputs
            total_loss = 0
            with tf.GradientTape() as tape:
                predictions = detection_model(images, training=True)
                loss = loss_fn.detection_loss(
                    predictions,
                    labels,
                    BATCH_SIZE,
                    416,
                    416,
                    1,
                    [
                        [[10, 13], [16, 30], [33, 32]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]],
                    ],
                )
                total_loss = loss
                gradients = tape.gradient(
                    total_loss, detection_model.trainable_variables
                )
                optimizer.apply_gradients(
                    zip(gradients, detection_model.trainable_variables)
                )

            return total_loss

        def test_step(inputs):
            images, labels = inputs
            predictions = detection_model(images, training=False)
            loss = det_loss.detection_loss(
                predictions,
                labels,
                BATCH_SIZE,
                416,
                416,
                1,
                [[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [59, 119]],
                [[10, 13], [16, 30], [33, 32]],
            )
            return loss

        @tf.function
        def distributed_train_step(data_set):
            per_replica_losses = mirrored_strategy.run(train_step, args=(data_set,))
            loss_over_replicas = mirrored_strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None
            )
            return loss_over_replicas

        @tf.function
        def distributed_test_step(dataset):
            return mirrored_strategy.run(test_step)

        epoch_template = "Epoch {}, Loss: {}"
        iter_template = "iter {}, Loss: {}"

        for epoch in range(NUM_EPOCHS):
            iteration = 0
            loss_per_epoch = 0
            for image, label in train_dataset:
                if batch_counter == BATCH_SIZE:
                    b_images, b_labels = dataloader.combine(batch_images, batch_labels)
                    loss = distributed_train_step(
                        (
                            b_images,
                            b_labels,
                        )
                    )
                    loss_per_epoch += loss
                    print(iter_template.format(iteration, loss))
                    batch_images = [image]
                    batch_labels = [label]
                    batch_counter = 1
                    iteration += 1
                elif batch_counter <= BATCH_SIZE:
                    batch_images.append(image)
                    batch_labels.append(label)
                    batch_counter += 1
                else:
                    print("Hitting Else")
            print(epoch_template.format(epoch, loss_per_epoch / iteration))
            saved_model_dir = "/home/karan/Checkpoint/ckpt"
            saver.save(saved_model_dir)
            # detection_model.save(saved_model_dir)
    # ---------------------- END OF DISTRIBUTED TRAINING -------------------------#

    # --------------------- START OF MANUAL TRAINING -------------------------#
    # buffer_size = 1000

    # batch_counter = 0
    # batch_images = []
    # batch_labels = []

    # detection_model = Detector(parsed_config, 1)
    # detection_model = detection_model.build_model((416, 416, 3))
    # # detection_model.summary()
    # detection_model.layers[0].trainable = False
    # # for layer in detection_model.layers:
    # #     print(layer.name, layer.trainable)
    # # exit()

    # # optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.8)
    # optimizer = tf.keras.optimizers.Adam()
    # saver = tf.train.Checkpoint(optimizer=optimizer, model=detection_model)

    # def train_step(inputs):
    #     images, labels = inputs
    #     total_loss = 0
    #     with tf.GradientTape() as tape:
    #         predictions = detection_model(images, training=True)
    #         loss = loss_fn.detection_loss(
    #             predictions,
    #             labels,
    #             BATCH_SIZE,
    #             416,
    #             416,
    #             1,
    #             [
    #                 [[116, 90], [156, 198], [373, 326]],
    #                 [[30, 61], [62, 45], [59, 119]],
    #                 [[10, 13], [16, 30], [33, 32]],
    #             ],
    #         )
    #         total_loss = loss  # sum([loss['total_loss'] for l in loss])
    #         gradients = tape.gradient(total_loss, detection_model.trainable_variables)
    #         optimizer.apply_gradients(
    #             zip(gradients, detection_model.trainable_variables)
    #         )

    #     return total_loss

    # def test_step(inputs):
    #     images, labels = inputs
    #     predictions = detection_model(images, training=False)
    #     loss = det_loss.detection_loss(
    #         predictions,
    #         labels,
    #         BATCH_SIZE,
    #         416,
    #         416,
    #         1,
    #         [
    #             [[10, 13], [16, 30], [33, 32]],
    #             [[30, 61], [62, 45], [59, 119]],
    #             [[116, 90], [156, 198], [373, 326]],
    #         ],
    #     )
    #     return loss  # sum([l['total_loss'] for l in loss])

    # epoch_template = "Epoch {}, Loss: {}"
    # iter_template = "iter {}, Loss: {}"

    # for epoch in range(NUM_EPOCHS):
    #     iteration = 0
    #     loss_per_epoch = 0
    #     for image, label in train_dataset:
    #         if batch_counter == BATCH_SIZE:
    #             b_images, b_labels = dataloader.combine(batch_images, batch_labels)
    #             print(b_images[0, :2, :2, :])
    #             loss = train_step(
    #                 (
    #                     b_images,
    #                     b_labels,
    #                 )
    #             )
    #             loss_per_epoch += loss
    #             print(iter_template.format(iteration, loss))

    #             batch_images = [image]
    #             batch_labels = [label]
    #             batch_counter = 1
    #             iteration += 1
    #         elif batch_counter <= BATCH_SIZE:
    #             batch_images.append(image)
    #             batch_labels.append(label)
    #             batch_counter += 1
    #         else:
    #             print("Hitting Else")
    #     print(epoch_template.format(epoch, loss_per_epoch / iteration))
    #     saved_model_dir = "/home/karan/Checkpoint/ckpt"
    #     saver.save(saved_model_dir)

# ------------------ END OF MANUAL TRAINING -------------------------#

# sample_image = np.random.randint(0, 255, size=(1, 416, 416, 3), dtype=np.uint8)
# input_image = np.array(sample_image / 255, dtype=np.float32)
# input_tensor = tf.convert_to_tensor(input_image)

# detection_model = Detector(parsed_config, 1)
# detection_model = detection_model.build_model(input_image.shape[1:])

# detection_model.summary()
# detection_out = detection_model(input_tensor)
# print(detection_out)

# for out_index, out in enumerate(detection_out):
#     with open(
#         os.path.join(
#             "/home/knapanda/local/object_detection/checkpoints/out_{}.npy".format(
#                 out_index
#             )
#         ),
#         "wb",
#     ) as f:
#         np.save(f, out)

# # save model
# saved_model_dir = (
#     "/home/knapanda/local/object_detection/checkpoints/saved_model/my_model/"
# )
# detection_model.save(saved_model_dir)


# DONT DO
# convert model to trt
# trt_model_dir = "/home/knapanda/local/object_detection/checkpoints/trt/"
# params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP16")
# converter = tf.experimental.tensorrt.Converter(
#     input_saved_model_dir=saved_model_dir, conversion_params=params
# )
# converter.convert()
# converter.save(trt_model_dir)
