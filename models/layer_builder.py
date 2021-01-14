from protos import convolution_pb2, residual_pb2, route_pb2, upsample_pb2, \
        yolo_pb2, network_pb2
from .layers import ConvolutionLayer, ResidualBlock, RouteLayer, \
        UpSamplingLayer, YoloLayer


def parse(cfg):
    if isinstance(cfg, convolution_pb2.Convolution):
        return parse_convolution_proto(cfg)
    elif isinstance(cfg, residual_pb2.ResidualBlock):
        return parse_residual_proto(cfg)
    elif isinstance(cfg, route_pb2.Route):
        return parse_route_proto(cfg)
    elif isinstance(cfg, upsample_pb2.UpSample):
        return parse_upsample_proto(cfg)
    elif isinstance(cfg, yolo_pb2.Yolo):
        return parse_yolo_proto(cfg)
    else:
        print("Recognized unknown layer type {}".format(isinstance(cfg)))
        exit()


def parse_convolution_proto(cfg):
    num_filters = int(cfg.num_filters)
    kernel_h = int(cfg.kernel_h)
    kernel_w = int(cfg.kernel_w)
    stride_h = int(cfg.stride_h)
    stride_w = int(cfg.stride_w)
    activation_fn = cfg.activation_function
    return ConvolutionLayer.ConvolutionLayer(num_filter=num_filters,
                                             kernel_size=(kernel_h, kernel_w),
                                             strides=(stride_h, stride_w),
                                             activation_int=activation_fn)


def parse_residual_proto(cfg):
    residual_multiplier = int(cfg.multiplier)
    residual_block_contents = []
    for _ in range(residual_multiplier):
        residual_block_contents.append([parse_convolution_proto(conv_cfg) for conv_cfg in cfg.bare_convs])
    return ResidualBlock(residual_block_contents, multiplier)

def parse_route_proto(cfg):
    input_indices = list(cfg.indices)
    #return input_indices
    return RouteLayer(input_indices)


def parse_upsample_proto(cfg):
    upsampling_factor = cfg.factor
    return UpSamplingLayer(upsampling_factor)


def parse_yolo_proto(cfg):
    anchors_cfg = list(cfg.anchors)
    len_anchors_cfg = len(anchors_cfg)
    assert len(len_anchors_cfg % 2 == 0, \
                "Odd number ofarguments {}".format(len_anchors_cfg)

    anchors_list = [
        [anchors_cfg[anchor_index*2], anchors_cfg[(anchor_index*2) + 1]]
        for anchor_index in range(int(len_anchors_cfg/2))
    ]
    num_classes = int(cfg.num_classes)
    num = int(cfg.num)
    jitter = float(cfg.jitter)
    ignore_threshold = float(cfg.ignore_threshold)
    truth_threshold = float(cfg.truth_threshold)
    random = float(cfg.float)
    return YoloLayer(anchors_list,
                     num_classes,
                     num,
                     jitter,
                     ignore_threshold,
                     truth_threshold,
                     random)

