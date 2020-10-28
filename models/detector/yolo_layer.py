from tensorflow.keras.layers import Conv2D


def yolo_layer(num_classes=1,
               anchors=[],
               name="detection"):
    num_channels = (5 + num_classes) * len(anchors)
    return Conv2D(num_channels,
                  1,
                  (1, 1),
                  padding="same",
                  data_format="channels_last",
                  name=name)
