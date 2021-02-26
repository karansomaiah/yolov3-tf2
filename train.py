#!/usr/bin/env python
"""
Script defining training functionality for the model.
Contains a trainer method which triggers any of the following:
    1. Training
    2. Training + Evaluation
    3. Evaluation
"""
import os
import tensorflow as tf

import enum

from models.model_builder_v2 import build

from models.loss.detection_loss_v2 import getLoss as DetectionLoss
from models import optimizer_builder
from data import dataloader_v2


class MODE(enum.Enum):
    TRAIN = 1
    TRAINEVAL = 2
    EVAL = 3


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


def trainer(config, is_training, is_evaluating):
    # init values
    if is_training:
        mode = MODE.TRAIN
        if is_evaluating:
            mode = MODE.TRAINEVAL
    elif is_evaluating:
        model = MODE.EVAL
    else:
        assert "Training Mode not supported."

    image_height = config.model_specs.image_height
    image_width = config.model_specs.image_width
    num_classes = config.model_specs.num_classes
    batch_size_train = config.train_config.batch_size
    batch_size_eval = config.eval_config.batch_size
    input_shape_without_batch = (image_height, image_width, 3)
    num_epochs = config.train_config.epochs

    # get respective dataset
    class_map = dataloader_v2.generate_class_map(config.class_def)
    train_data = dataloader_v2.build(
        config.train_config.dataset,
        class_map,
        image_height,
        image_width,
        batch_size_train,
    )  # .repeat(100)
    eval_data = dataloader_v2.build(
        config.eval_config.dataset,
        class_map,
        image_height,
        image_width,
        batch_size_eval,
    )  # .repeat(10)

    # define the strategy
    strategy = tf.distribute.MirroredStrategy()

    # initialize the model here
    with strategy.scope():
        if mode == MODE.TRAIN:
            model, inputs, outputs, anchors = build(
                config, input_shape_without_batch, batch_size_train
            )
            assert (
                train_data and not eval_data
            ), "train_data should exist without eval_data."

        elif mode == MODE.TRAINEVAL:
            model, inputs, outputs, anchors = build(
                config, input_shape_without_batch, batch_size_train
            )
            assert (
                train_data and eval_data
            ), "train_data and eval_data should both exist."

        else:  # mode is only eval
            model, inputs, outputs, anchors = build(
                config, input_shape_without_batch, batch_size_train
            )
            assert (
                not train_data and eval_data
            ), "eval_data should exist without train_data"

        # set the image/input layer as non-trainable
        print(model.summary())
        #with open('training_logs/model_info.txt', 'w') as f:
        #    for layer in model.layers:
        #        f.write("Layer name: {}\n".format(layer.name))
        #        f.write("all weights:\n")
        #        for weight in layer.weights:
        #            f.write(weight.name + "\n")
        #        f.write("trainable_weights\n")
        #        for weight in layer.trainable_weights:
        #            f.write(weight.name + "\n")
        #        f.write("\n")
        #    f.write("\n")

        loss_fn_anchors_large = DetectionLoss(
            batch_size_train,
            anchors[0],
            len(anchors[0]),
            image_width,
            image_height,
            top_k=100,
        )
        loss_fn_anchors_medium = DetectionLoss(
            batch_size_train,
            anchors[1],
            len(anchors[1]),
            image_width,
            image_height,
            top_k=100,
        )

        loss_fn_anchors_small = DetectionLoss(
            batch_size_train,
            anchors[2],
            len(anchors[2]),
            image_width,
            image_height,
            top_k=100,
        )

        # get the optimizer
        optimizer = optimizer_builder.build(config.train_config.optimizer_config)

        model.compile(
            loss_fn1=loss_fn_anchors_large,
            loss_fn2=loss_fn_anchors_medium,
            loss_fn3=loss_fn_anchors_small,
            optimizer=optimizer,
            run_eagerly=True,
        )

        if not os.path.isdir(config.train_config.log):
            os.makedirs(config.train_config.log)

        # Callback for printing the LR at the end of each epoch.
        class PrintLR(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(
                    "\nLearning rate for epoch {} is {}".format(
                        epoch + 1, model.optimizer.lr.numpy()
                    )
                )

        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(config.train_config.log, "tensorboad")
            ),
            tf.keras.callbacks.ModelCheckpoint(filepath=config.train_config.log),
            PrintLR(),
        ]

        model.fit(
            x=train_data,
            epochs=num_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=eval_data,
        )

    # @To-Do: Checkpoint loader to be a separate thing.
    # @To-Do: Add capability to load Darknet weights.
    # loading pretrained models can be done here.
    # if config_pb.train_config.pretrained:
    #     checkpoint = tf.train.Checkpoint(model=model)
    #     if 'ckpt' in config_pb.train_config.fine_tune_checkpoint:
    #         latest_checkpoint = config_pb.train_config.pretrained
    #     else:
    #         latest_checkpoint = tf.train.latest_checkpoint(config_pb.train_config.pretrained)
    #     checkpoint.restore(latest_checkpoint)


# def train_and_evaluate(
#    config,
#    detection_model,
#    mode,
#    optimizer,
#    image_height,
#    image_width,
#    train_batch_size,
#    eval_batch_size,
#    train_dataset,
#    eval_dataset,
#    class_map,
#    num_epochs,
#    checkpointer,
#    **kwargs
# ):
#    # initialize for training
#    mirrored_strategy = tf.distribute.MirroredStrategy()
#    with mirrored_strategy.scope():
#        # single train step given inputs
#        def train_step(inputs):
#            images, labels = inputs
#            total_loss = 0
#            with tf.GradientTape() as tape:
#                predictions = detection_model(images, training=True)
#                loss = loss_module.detection_loss(
#                    predictions,
#                    labels,
#                    train_batch_size,
#                    image_height,
#                    image_width,
#                    num_classes,
#                    detection_model.anchors,
#                    # [
#                    #    [[10, 13], [16, 30], [33, 32]],
#                    #    [[30, 61], [62, 45], [59, 119]],
#                    #    [[116, 90], [156, 198], [373, 326]],
#                    # ],
#                )
#                total_loss = loss
#                gradients = tape.gradient(
#                    total_loss, detection_model.trainable_variables
#                )
#                optimizer.apply_gradients(
#                    zip(gradients, detection_model.trainable_variables)
#                )
#            return total_loss
#
#        # single test step
#        def eval_step(inputs):
#            images, labels = inputs
#            predictions = detection_model(images, training=False)
#            loss = det_loss.detection_loss(
#                predictions,
#                labels,
#                eval_batch_size,
#                image_height,
#                image_width,
#                num_classes,
#                detection_model.anchors,
#                # [[116, 90], [156, 198], [373, 326]],
#                # [[30, 61], [62, 45], [59, 119]],
#                # [[10, 13], [16, 30], [33, 32]],
#            )
#            return loss
#
#        # distributed train step
#        @tf.function
#        def distributed_train_step(dataset):
#            per_replica_losses = mirrored_strategy.run(train_step, args=(dataset,))
#            loss_over_replicas = mirrored_strategy.reduce(
#                tf.distribute.ReduceOp.MEAN, per_replica_losses
#            )
#            return loss_over_replicas
#
#        # distributed test step
#        @tf.function
#        def distributed_eval_step(dataset):
#            return mirrored_strategy.run(test_step)
#
#        # metric/loss logging template.
#        epoch_template = (
#            "epoch :{}/{}      train_loss: {}      eval_loss:{}\n"
#            "                  mAP: {}\n"
#        )
#        for class_name, class_id in class_map.items():
#            epoch_template += (
#                "                  mAP for class {} ".format(class_name) + ":{}\n"
#            )
#
#        iter_template = "epoch {} | iter: {}/{}    loss:{}"
#
#        # find number of iterations
#        for _i, (_img, _val) in enumerate(train_dataset):
#            pass
#        train_dataset_length = _i + 1
#        for _i, (_img, _val) in enumerate(validation_dataset):
#            pass
#        validation_dataset_length = _i + 1
#        # training loop
#        for epoch in range(1, epochs + 1):
#            # variables for each epoch
#            per_epoch_cumulative_loss = 0.0
#            for iter_num, (images, labels) in train_dataset:
#                loss = distributed_train_step((images, labels))
#                per_epoch_cumulative_loss += loss
#                # print per iteration loss
#                iter_template.format(epoch, iter_num + 1, train_dataset_length, loss)
#                print(iter_template)
#            # completion of an iteration
#            avg_loss = (per_epoch_average_loss) / train_dataset_length
#            # get metrics
#            eval_loss, eval_map, classwise_map = get_metrics()
#            # print per epoch loss
#            epoch_template.format()
#
#    return 1
#
#
# def train_only(config, model, **kwargs):
#    return 1
#
#
# def train_step():
#    return 1
