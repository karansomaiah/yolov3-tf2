#!/usr/bin/env python
class Layer(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.args = args
        self.attributes = {}

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Layer):
            return NotImplemented
        return self.name == other.name

    def info(self):
        print("Layer Name: {}".format(self.name))

        if len(self.attributes) > 0:
            for key, value in self.attributes.items():
                print("Attribute -> {}: {}".format(key, value))
        elif len(self.kwargs) > 0 or len(self.args) > 0:
            if len(self.kwargs) > 0:
                for key, value in self.kwargs.items():
                    print("kwarg -> {}: {}".format(key, value))
            if len(self.args) > 0:
                for value in self.args:
                    print("arg -> {}".format(value))
        else:
            print("Empty Layer!")


class Convolution(Layer):
    """
    Usage Convolution(name,
                      num_kernels=1,
                      kernel_h=1,
                      kernel_w=1,
                      stride_h=1,
                      stride_w=1,
                      activation="relu", # one of relu,lrelu,etc.
                      bn=True)
    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.num_kernels = 0
        self.kernel_h = 0
        self.kernel_w = 0
        self.stride_h = 0
        self.stride_w = 0
        self.activation = "relu"
        self.batch_norm = True
        self.mapping = {
            "num_kernels": self.num_kernels,
            "kernel_h": self.kernel_h,
            "kernel_w": self.kernel_w,
            "stride_h": self.stride_h,
            "stride_w": self.stride_w,
            "activation": self.activation,
            "bn": self.batch_norm,
        }
        super(Convolution, self).__init__(name, *args, **kwargs)
        self.set()

    def set(self):
        # ignoring *args here
        for key, value in self.kwargs.items():
            if key.upper().lower() in self.mapping.keys():
                self.mapping[key.upper().lower()] = value
        self.attributes = self.mapping


class Residual(Layer):
    """
    Call as follows:

    Residual("residual",
             pattern=[Convolution, Convolution, Connection],
             repeat=2)
    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.pattern = []
        self.repeat = 1
        self.mapping = {"pattern": self.pattern, "repeat": self.repeat}
        super(Residual, self).__init__(name, *args, **kwargs)
        self.set()

    def set(self):
        # args are ignored
        temp_pattern = []

        for k, v in self.kwargs.items():
            if k == "pattern":  # check for pattern keyword
                assert (
                    type(v) == list
                ), "Argument provided to patter should be a list but got {}".format(
                    type(v)
                )
                for layer in v:
                    assert issubclass(
                        type(layer), Layer
                    ), "Arguments to pattern is a list but contents are not of type Layer"
                    temp_pattern.append(layer)

        if len(temp_pattern) > 0:  # check if pattern was filled
            if "repeat" not in self.kwargs.keys():
                self.pattern = temp_pattern
                self.mapping["pattern"] = self.pattern
            else:
                self.pattern = temp_pattern * self.kwargs["repeat"]
                self.repeat = self.kwargs["repeat"]

                self.mapping["pattern"] = temp_pattern * self.kwargs["repeat"]
                self.mapping["repeat"] = self.repeat

        self.attributes = self.mapping


class Connection(Layer):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(Connection, self).__init__(name, *args, **kwargs)
        self.set()

    def set(self):
        self.attributes = {}


class Route(Layer):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.layer_connection_indices = []
        super(Route, self).__init__(name, *args, **kwargs)
        self.set()

    def set(self):
        assert (
            len(self.args) > 0
        ), "Send some arguments into Connection Layer. Received {}".format(self.args)
        for index in self.args:
            self.layer_connection_indices.append(index)
        self.attributes = {"indices": self.layer_connection_indices}


class UpSample(Layer):
    """
    Usage UpSample("upsample",
                   factor=2)
    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.factor = 1
        super(UpSample, self).__init__(name, *args, **kwargs)
        self.set()

    def set(self):
        if "factor" in self.kwargs.keys():
            self.factor = self.kwargs["factor"]
        self.attributes = {"factor": self.factor}


class Yolo(Layer):
    """
    Usage Yolo(name,
               anchors=[],
               num_classes=1,
               num=1,
               jitter=0.0,
               ignore_thresh=0.0,
               truth_thresh=1,
               random=1)
    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.anchors = []
        self.num_classes = 1
        self.num = 1
        self.jitter = 0.0
        self.ignore_thresh = 0.0
        self.truth_thresh = 1
        self.random = 1
        self.mapping = {
            "anchors": self.anchors,
            "num_classes": self.num_classes,
            "num": self.num,
            "jitter": self.jitter,
            "ignore_thresh": self.ignore_thresh,
            "thruth_thresh": self.truth_thresh,
            "random": self.random,
        }
        super(Yolo, self).__init__(name, *args, **kwargs)
        self.set()

    def set(self):
        for key, value in self.kwargs.items():
            if key.upper().lower() in self.mapping.keys():
                self.mapping[key.upper().lower()] = value
        self.attributes = self.mapping
