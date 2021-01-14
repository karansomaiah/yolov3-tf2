#!/usr/bin/env python
"""
Script defining training functionality for the model.
Contains a trainer method which triggers any of the following:
    1. Training
    2. Training + Evaluation
    3. Evaluation
"""
import tensorflow as tf
import enum
from models.model_builder import DetectorModel
from models.loss import detection_loss as loss_module
from models import optimizer_builder
from data import dataloader
from eval import get_metrics


class MODE(enum.Enum):
    TRAIN = 1
    TRAINEVAL = 2
    EVAL = 3


def trainer(config, is_training, is_evaluating):
    # init values
    if is_training:
        if is_evaluating:
            mode = MODE.TRAINEVAL
        mode = MODE.TRAIN
    elif is_evaluating:
        model = MODE.EVAL
    else:
        assert "Training Mode not supported."

    image_height = config.model_specs.image_height
    image_width = config.model_specs.image_width
    num_classes = config.model_specs.num_classes
    batch_size_train = config.train_config.batch_size
    batch_size_eval = config.eval_config.batch_size
    input_shape_without_batch = (IM_H, IM_W, 3)
    num_epochs = config.train_config.epochs

    # get respective dataset
    class_map = dataloader.generate_class_map(config.class_def)
    train_data = dataloader.tf_dataloader_v2(
        config.train_config.dataset, class_map, image_height, image_width
    )
    eval_data = dataloader.tf_dataloader_v2(
        config.eval_config.dataset, class_map, image_height, image_width
    )

    # initialize the model here
    if mode == MODE.TRAIN:
        model = DetectorModel(config)
        model.build_model(input_shape_without_batch, batch_size_train)
        assert (
            train_data and not eval_data
        ), "train_data should exist without eval_data."

    elif mode == MODE.TRAINEVAL:
        model = DetectorModel(config)
        model.build_model(input_shape_without_batch, batch_size_train)
        assert train_data and eval_data, "train_data and eval_data should both exist."

    else:  # mode is only eval
        model = DetectorModel(config)
        model.build_model(input_shape_without_batch, batch_size_eval)
        assert not train_data and eval_data, "eval_data should exist without train_data"

    # set the image/input layer as non-trainable
    model.layers[0].trainable = False

    # get the optimizer
    optimizer = optimizer_builder.build(config_pb.train_config.optimizer_config)
    if mode != MODE.EVAL:
        saver = tf.train.Checkpoint(optimizer=optimizer, model=model)
    else:
        saver = tf.train.Checkpoint(model=model)

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


def train_and_evaluate(
    config,
    detection_model,
    mode,
    optimizer,
    image_height,
    image_width,
    train_batch_size,
    eval_batch_size,
    train_dataset,
    eval_dataset,
    class_map,
    num_epochs,
    checkpointer,
    **kwargs
):
    # initialize for training
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # single train step given inputs
        def train_step(inputs):
            images, labels = inputs
            total_loss = 0
            with tf.GradientTape() as tape:
                predictions = detection_model(images, training=True)
                loss = loss_module.detection_loss(
                    predictions,
                    labels,
                    train_batch_size,
                    image_height,
                    image_width,
                    num_classes,
                    detection_model.anchors,
                    # [
                    #    [[10, 13], [16, 30], [33, 32]],
                    #    [[30, 61], [62, 45], [59, 119]],
                    #    [[116, 90], [156, 198], [373, 326]],
                    # ],
                )
                total_loss = loss
                gradients = tape.gradient(
                    total_loss, detection_model.trainable_variables
                )
                optimizer.apply_gradients(
                    zip(gradients, detection_model.trainable_variables)
                )
            return total_loss

        # single test step
        def eval_step(inputs):
            images, labels = inputs
            predictions = detection_model(images, training=False)
            loss = det_loss.detection_loss(
                predictions,
                labels,
                eval_batch_size,
                image_height,
                image_width,
                num_classes,
                detection_model.anchors,
                # [[116, 90], [156, 198], [373, 326]],
                # [[30, 61], [62, 45], [59, 119]],
                # [[10, 13], [16, 30], [33, 32]],
            )
            return loss

        # distributed train step
        @tf.function
        def distributed_train_step(dataset):
            per_replica_losses = mirrored_strategy.run(train_step, args=(dataset,))
            loss_over_replicas = mirrored_strategy.reduce(
                tf.distribute.ReduceOp.MEAN, per_replica_losses
            )
            return loss_over_replicas

        # distributed test step
        @tf.function
        def distributed_eval_step(dataset):
            return mirrored_strategy.run(test_step)

        # metric/loss logging template.
        epoch_template = (
            "epoch :{}/{}      train_loss: {}      eval_loss:{}\n"
            "                  mAP: {}\n"
        )
        for class_name, class_id in class_map.items():
            epoch_template += (
                "                  mAP for class {} ".format(class_name) + ":{}\n"
            )

        iter_template = "epoch {} | iter: {}/{}    loss:{}"

        # find number of iterations
        for _i, (_img, _val) in enumerate(train_dataset):
            pass
        train_dataset_length = _i + 1
        for _i, (_img, _val) in enumerate(validation_dataset):
            pass
        validation_dataset_length = _i + 1
        # training loop
        for epoch in range(1, epochs + 1):
            # variables for each epoch
            per_epoch_cumulative_loss = 0.0
            for iter_num, (images, labels) in train_dataset:
                loss = distributed_train_step((images, labels))
                per_epoch_cumulative_loss += loss
                # print per iteration loss
                iter_template.format(epoch, iter_num + 1, train_dataset_length, loss)
                print(iter_template)
            # completion of an iteration
            avg_loss = (per_epoch_average_loss) / train_dataset_length
            # get metrics
            eval_loss, eval_map, classwise_map = get_metrics()
            # print per epoch loss
            epoch_template.format()

    return 1


def train_only(config, model, **kwargs):
    return 1


def train_step():
    return 1
