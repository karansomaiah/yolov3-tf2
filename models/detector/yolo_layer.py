from tensorflow.keras.layers import Conv2D, Reshape


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

def structure_yolo_output(input,
                          batch_size=1,
                          anchors=[],
                          grid_x_size=1,
                          grid_y_size=1,
                          num_classes=1):
    return Reshape((batch_size, anchors, grid_x_size, grid_y_size, num_classes + 5))
