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
from models.callbacks import LoggingCallback
from eval_metrics import Evaluate, EvalType


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
    )
    eval_data = dataloader_v2.build(
        config.eval_config.dataset,
        class_map,
        image_height,
        image_width,
        batch_size_eval,
    )

    # define the strategy
    strategy = tf.distribute.MirroredStrategy()
    evaluator = Evaluate(
        labels_list=['Dog'], 
        ious_to_report=[0.5, 0.75], 
        eval_type=EvalType.trainVal
    )

    # initialize the model here
    with strategy.scope():
        if mode == MODE.TRAIN:
            model, inputs, outputs, anchors = build(
                config, input_shape_without_batch, batch_size_train, evaluator
            )
            assert (
                train_data and not eval_data
            ), "train_data should exist without eval_data."

        elif mode == MODE.TRAINEVAL:
            model, inputs, outputs, anchors = build(
                config, input_shape_without_batch, batch_size_train, evaluator
            )
            assert (
                train_data and eval_data
            ), "train_data and eval_data should both exist."

        else:  # mode is only eval
            model, inputs, outputs, anchors = build(
                config, input_shape_without_batch, batch_size_eval, evaluator
            )
            assert (
                not train_data and eval_data
            ), "eval_data should exist without train_data"


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
            loss_fn_large=loss_fn_anchors_large,
            loss_fn_medium=loss_fn_anchors_medium,
            loss_fn_small=loss_fn_anchors_small,
            optimizer=optimizer,
            run_eagerly=True,
        )


        # summary writer
        tensorboard_logging_dir = os.path.join(config.train_config.log, "tensorboard")
        if not os.path.isdir(tensorboard_logging_dir):
            os.mkdir(tensorboard_logging_dir)
        
        train_tb_logdir = os.path.join(tensorboard_logging_dir, "train")
        if not os.path.isdir(train_tb_logdir):
            os.mkdir(train_tb_logdir)

        eval_tb_logdir = os.path.join(tensorboard_logging_dir, "validation")
        if not os.path.isdir(eval_tb_logdir):
            os.mkdir(eval_tb_logdir)

        train_file_writer = tf.summary.create_file_writer(train_tb_logdir)
        eval_file_writer = tf.summary.create_file_writer(eval_tb_logdir)

        checkpoint_path = os.path.join(config.train_config.log, "model.ckpt")
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if not os.path.isdir(config.train_config.log):
            os.makedirs(config.train_config.log)

        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir="training_logs/tensorboard"
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                verbose=1
            ),
            LoggingCallback(
                train_writer=train_file_writer,
                validation_writer=eval_file_writer
            )
        ]

        model.fit(
            x=train_data,
            epochs=num_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=eval_data,
        )

