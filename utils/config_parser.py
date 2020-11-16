import os
import ast
from utils import *

# import network_placeholders

# from network_placeholders import *


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


def read_cfg_by_line(filepath):
    with open(filepath, "r") as filereader:
        lines = filereader.readlines()

    return [line[:-1] for line in lines]


def parse_cfg(lines):
    """
    Define parsing functions to get keyword arguments to specific layers and call those layers
    """

    def parse_convolution(text):
        elements = text.strip().split(",")
        assert len(elements) == 6, "Irregular Convolution Definition. {}".format(
            elements
        )

        # start parsing
        name = elements[0]
        num_filters = int(elements[1])
        kernel_h = int(elements[2])
        kernel_w = int(elements[3])
        stride_h = int(elements[4])
        stride_w = int(elements[5])

        return Convolution(
            name,
            num_kernels=num_filters,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
        )

    def parse_connection(text):
        elements = text.strip().split(",")
        assert len(elements) == 1, "Irregular Connection Definition. {}".format(
            elements
        )

        # start parsing
        name = elements[0]
        return Connection(name)

    def parse_yolo(text):
        # find brackets
        bracket_start = 0
        bracket_end = bracket_start + 1
        found = False
        for el_i, element in enumerate(text):
            if "[" == element:
                bracket_start = el_i
            elif "]" == element:
                bracket_end = el_i
                found = True
                break
            else:
                continue

        assert found, "Yolo layer format is incorrect. {}".format(text)

        name = text[: bracket_start - 1]
        anchors = ast.literal_eval(text[bracket_start : bracket_end + 1])
        elements = text[bracket_end + 2 :].strip().split(",")

        assert len(elements) == 6, "Irregular yolo layer arguments. {}".format(elements)
        num_classes = int(elements[0])
        num = int(elements[1])
        jitter = float(elements[2])
        ignore_thresh = float(elements[3])
        truth_thresh = int(elements[4])
        random = int(elements[5])

        return Yolo(
            name,
            anchors=anchors,
            num_classes=num_classes,
            num=num,
            jitter=jitter,
            ignore_thresh=ignore_thresh,
            truth_thresh=truth_thresh,
            random=random,
        )

    def parse_route(text):
        elements = text.strip().split(",")
        assert len(elements) > 0, "Irregular Route definition. {}".format(text)

        name = elements[0]

        if len(elements) > 1:
            route_inputs = [int(el) for el in elements[1:]]
        else:
            route_inputs = [-1]

        return Route(name, *route_inputs)

    def parse_upsample(text):
        elements = text.strip().split(",")
        assert len(elements) > 1, "Irregular Upsample Definition"

        name = elements[0]
        factor = int(elements[1])
        return UpSample(name, factor=factor)

    def parse_residual(list_of_text):
        """Potential input will be
        residual,2
            convolution,64,3,3,1,1
            convolution,32,1,1,1,1
            connection
        """
        assert type(list_of_text) == list, "Input isn't a List {}".format(list_of_text)

        residual_line = list_of_text[0]
        residual_name, residual_factor = residual_line.strip().split(",")

        residual_elements = list_of_text[1:]
        layers = []
        for layer_line in residual_elements:
            layer_args = layer_line.strip().split(",")
            parser = parser_mapping[layer_args[0]]
            layer = parser(layer_line.strip())
            layers.append(layer)

        return Residual(residual_name, pattern=layers, repeat=int(residual_factor))

    parser_mapping = {
        "convolution": parse_convolution,
        "residual": parse_residual,
        "connection": parse_connection,
        "yolo": parse_yolo,
        "route": parse_route,
        "upsample": parse_upsample,
    }

    Layer_List = []
    Layer_Index = 0

    current_index = 0
    while current_index < len(lines):
        line = lines[current_index]
        keyword = line.split(",")[0]

        # check if residual
        residual_inputs = []
        if keyword == "residual":
            residual_inputs.append(line)
            temp_index = current_index
            found = False
            while not found:
                element_line = lines[temp_index + 1]
                element_line_split = element_line.split(",")
                if " " in element_line_split[0]:
                    residual_inputs.append(lines[temp_index + 1])
                    temp_index += 1
                else:
                    current_index = temp_index + 1
                    found = True
                    break

            Layer_List.append(parser_mapping[keyword](residual_inputs))
        else:
            Layer_List.append(parser_mapping[keyword](line))
            current_index += 1

    return Layer_List


def expand_ops(list_of_layers):

    ops = []

    layer_count = 0
    for layer in list_of_layers:

        residual_layer_expanded = []
        if type(layer) == Residual:
            heading = (-1, "residual")
            residual_layer_expanded.append(heading)
            # only on entry the pre will be the previous conv, rest is all the connection layer
            layer_pre = ops[-1]

            for residual_part in layer.pattern:
                if type(residual_part) == Connection:
                    layer_count += 1
                    layer_post = residual_layer_expanded[-1]
                    residual_layer_expanded.append(
                        (layer_count, [residual_part, layer_pre[0], layer_post[0]])
                    )
                    layer_pre = residual_layer_expanded[-1]
                else:
                    layer_count += 1
                    residual_layer_expanded.append((layer_count, residual_part))

            ops += residual_layer_expanded

        elif type(layer) == Route:
            layer_count += 1
            routes = [
                layer_count + back_index
                for back_index in layer.layer_connection_indices
            ]
            ops.append((layer_count, [layer] + routes))
        else:
            layer_count += 1
            ops.append((layer_count, layer))

    return ops


def parse(filepath):
    return expand_ops(parse_cfg(read_cfg_by_line(filepath)))


if __name__ == "__main__":
    cfg_filepath = "/home/knapanda/local/casual/yolov3-tf2/cfg/darknet.cfg"
    cfg_lines = read_cfg_by_line(cfg_filepath)

    parsed_cfg = parse_cfg(cfg_lines)
    expanded_ops = expand_ops(parsed_cfg)

    Conv1 = Convolution("conv1")
    Conv2 = Convolution("conv2")
    print(Conv1 == Conv2)

    # print(expanded_ops)
    # print(type(expanded_ops[0][0]), type(expanded_ops[0][1]))

    # for op_index, op in expanded_ops:
    #     print("{} -> {}".format(op_index, op))
