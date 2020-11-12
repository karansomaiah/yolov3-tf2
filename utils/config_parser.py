import os
import ast
from utils import *
from network_placeholders import *


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


if __name__ == "__main__":
    cfg_filepath = "/home/knapanda/local/casual/yolov3-tf2/cfg/darknet.cfg"
    cfg_lines = read_cfg_by_line(cfg_filepath)

    parsed_cfg = parse_cfg(cfg_lines)
    expanded_ops = expand_ops(parsed_cfg)
    print(expanded_ops)

    for op_index, op in expanded_ops:
        print("{} -> {}".format(op_index, op))
