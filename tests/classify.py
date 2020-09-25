#! python3
""" Testing the Darknet backbone for classification
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

sys.path.append("../")
from models.backbone.darknet53 import DarkNet53

NUM_CLASSES = 102
BATCH_SIZE = 10
NUM_EPOCHS = 100


def build_classification_model():
    darknet_backbone = DarkNet53(NUM_CLASSES)
    darknet_backbone.build_graph(
        (
            BATCH_SIZE,
            128,
            128,
            3,
        )
    )

    print(darknet_backbone.summary())
    sample_image = np.asfarray(np.random.randint(0, 255, (10, 256, 256, 3), np.uint8))
    print("Printing forward pass time")
    start_time = time.time()
    sample_output = darknet_backbone(sample_image)
    print("sample output {}".format(sample_output))
    end_time = time.time()
    print("Time taken to conduct inference: {}".format(end_time - start_time))
    return darknet_backbone


def load_dataset():
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
        1.0 / 255
    )
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "C:\\Users\karan\\tensorflow_datasets\\downloads\\101_ObjectCategories\\",
        validation_split=0.2,
        subset="training",
        seed=123,
        batch_size=1,
    )
    train_ds_norm = train_ds.map(lambda x, y: (normalization_layer(x), y))
    train_ds_norm.cache()
    train_ds_norm.shuffle(buffer_size=1000)
    train_ds_norm.batch(10)
    train_ds_norm.prefetch(tf.data.experimental.AUTOTUNE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "C:\\Users\karan\\tensorflow_datasets\\downloads\\101_ObjectCategories\\",
        validation_split=0.2,
        subset="validation",
        seed=123,
        batch_size=1,
    )
    val_ds_norm = val_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds_norm.cache()
    val_ds_norm.shuffle(buffer_size=1000)
    val_ds_norm.batch(10)
    val_ds_norm.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds


def run_training(train_dataset, validation_dataset, model):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    print("About to train....")

    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #     train_loss(loss)
    #     train_accuracy(labels, predictions)

    # @tf.function
    # def test_step(images, labels):
    #     predictions = model(images, training=False)
    #     t_loss = loss_object(labels, predictions)

    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)

    EPOCHS = NUM_EPOCHS

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        # training step
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(labels, predictions)

        # testing step
        for test_images, test_labels in validation_dataset:
            test_predictions = model(test_images, training=False)
            t_loss = loss_object(test_labels, test_predictions)
            test_loss(t_loss)
            test_accuracy(test_labels, test_predictions)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )


if __name__ == "__main__":
    classifier = build_classification_model()
    # 6train_data, validation_data = load_dataset()
    # run_training(train_data, validation_data, classifier)