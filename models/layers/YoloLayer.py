from tensorflow.keras.layers import Conv2D, Reshape, Layer


class YoloLayer(Layer):
    """Yolo Detection Layer."""

    def __init__(
        self,
        anchors,
        num_classes=1,
        num=3,
        jitter=0.3,
        ignore_threshold=0.7,
        truth_threshold=1.0,
        random=1,
        **kwargs
    ):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.num = num
        self.jitter = jitter
        self.ignore_threshold = ignore_threshold
        self.truth_threshold = truth_threshold
        self.num_channels = num_channels = (5 + num_classes) * len(anchors)
        self.yolo_layer = Conv2D(
            filters=num_channels,
            kernel_size=1,
            strides=(1, 1),
            padding="same",
            data_format="channels_last",
        )

    def call(self, inputs):
        return self.yolo_layer(inputs)
