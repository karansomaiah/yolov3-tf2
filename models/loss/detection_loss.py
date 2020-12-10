#!/usr/bin/env python
"""
Set detection losses for outputs
"""

import tensorflow as tf
import tensorflow.keras as keras
# from losses import *
from utils.box_utils import box_iou_tf
from utils.utils import INT, FLOAT, BOOL


current_loss_map: {
    'mse' : keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
    'bce' : keras.losses.BinaryCrossentropy()
}


def get_grid_shapes(detections):
    return [
       (detection.shape[1], detection.shape[2]) for detection in detections
    ]


def get_grid_offsets(grid_x_num, grid_y_num):
    """
    """
    intermediate_center_x_offset = tf.broadcast_to(
        tf.range(grid_x_num, dtype=tf.float32),
        [grid_y_num, grid_x_num]
    )
    center_x_offsets = tf.broadcast_to(
        intermediate_center_x_offset,
        [batch_size, grid_y_num, grid_x_num]
    )
    intermediate_center_y_offset = tf.broadcast_to(
        tf.range(grid_y_num, dtype=tf.float32),
        [grid_x_num, grid_y_num]
    )
    intermediate_center_y_offset = tf.transpose(intermediate_center_y_offset)
    center_y_offsets = tf.broadcast_to(
        intermediate_center_y_offset,
        [batch_size, grid_y_num, grid_x_num]
    )
    return center_x_offsets, center_y_offsets


def get_anchor_wh_offsets(w_index,
                          h_index,
                          anchors,
                          grid_x,
                          grid_y,
                          batch_size):
    anchor_size = anchors.shape[0]
    w_offsets = tf.broadcast_to(
        tf.expand_dims(
            tf.broadcast_to(
                tf.expand_dims(
                    tf.broadcast_to(
                        tf.expand_dims(anchors[:, w_index], axis=-1), 
                        [anchor_size, grid_y]), 
                    axis=-1), 
                [anchor_size, grid_y, grid_x]), 
            axis=0), 
        [batch_size, anchor_size, grid_y, grid_x]
    )

    h_offsets = tf.broadcast_to(
        tf.expand_dims(
            tf.broadcast_to(
                tf.expand_dims(
                    tf.broadcast_to(
                        tf.expand_dims(anchors[:, h_index], axis=-1), 
                        [anchor_size, grid_y]), 
                    axis=-1), 
                [anchor_size, grid_y, grid_x]), 
            axis=0), 
        [batch_size, anchor_size, grid_y, grid_x]
    )
    return w_offsets, h_offsets


def convert_gt_for_loss(ground_truth,
                        prior_x,
                        prior_y,
                        matched_anchors,
                        image_width,
                        image_height,
                        grid_x,
                        grid_y):
    w_index = 0
    h_index = 1
    x_index, y_index, w_index, h_index = 0, 1, 2, 3

    ground_truth_x = (ground_truth[:, x_index] * grid_x) - prior_x
    ground_truth_y = (ground_truth[:, y_index] * grid_y) - prior_y
    ground_truth_w = tf.math.log(ground_truth[:, w_index] * (image_width/matched_anchors[:, w_index]))
    ground_truth_h = tf.math.log(ground_truth[:, h_index] * (image_height/matched_anchors[:, h_index]))

    return tf.stack(
        [ground_truth_x,
         ground_truth_y,
         ground_truth_w,
         ground_truth_h],
        axis=-1
    )


def get_localization_loss(true, pred):
    return current_loss_map['mse'](true, pred)


def get_noobj_loss(true, pred):
    return current_loss_map['bce'](true, pred)


def get_obj_loss(true, pred):
    return current_loss_map['bce'](true, pred)


def get_class_loss(true, pred):
    return current_loss_map['bce'](true, pred)


def get_loss_per_grid_detection(detection_output,
                                ground_truth,
                                anchors,
                                grid_shape,
                                image_wh,
                                batch_size,
                                num_classes):
    losses = {
        'regression_loss': 0.0,
        'objectness_loss': 0.0,
        'noobjectness_loss': 0.0,
        'classification_loss': 0.0,
        'total_loss': 0.0
    }

    num_anchors = len(anchors_list)
    grid_x, grid_y = grid_shape
    image_width, image_height = image_wh
    width_downsampling_factor = image_width/grid_x
    height_downsampling_factor = image_height/grid_y
    num_channels = detection_output.shape[-1]
    prediction_channels = num_channels/num_anchors
    num_gt = ground_truth.shape[0]

    # ground truth tensors
    ground_truth_bbox = ground_truth[:, 1:-1]
    ground_truth_wh_bbox = tf.concat(
        [
            tf.zeros_like(ground_truth_bbox[:, -2:]),
            ground_truth_bbox[:, -2:]
        ],
        axis=-1
    )
    ground_truth_classes = tf.cast(ground_truth[:, -1], dtype=INT)
    ground_truth_classes_onehot = tf.one_hot(ground_truth_classes, depth=num_classes) 
    ground_truth_batch_indices = tf.cast(ground_truth[:, 0], dtype=INT)
    ground_truth_x = ground_truth_bbox[:, 0]
    ground_truth_y = ground_truth_bbox[:, 1]
    ground_truth_x_scaled = ground_truth_x * grid_x
    ground_truth_y_scaled = ground_truth_y * grid_y

    # anchors tensor
    anchors_tensor = tf.constant(anchors, dtype=FLOAT)
    anchors_normalized = tf.math.divide(anchors_tensor,
                                        tf.broadcast_to(
                                            tf.expand_dims(
                                                tf.constant([float(image_width), float(image_height)], dtype=FLOAT), 
                                                axis=0),
                                            [anchors_tensor.shape[0], anchors_tensor.shape[1]]
                                        )
                                       )
    anchors_normalized_bbox = tf.concat(
        [tf.zeros_like(anchors_normalized),
         anchors_normalized],
        axis=-1
    )

    # x & y - offsets
    center_x, center_y = get_grid_offsets(grid_x,
                                          grid_y)
    
    # reshape output tensor to [batch_size, 
    #                           anchor_size, 
    #                           grid_x, 
    #                           grid_y, 
    #                           pred_channels]
    output_reshaped = tf.reshape(detection_output,
                                 [batch_size, 
                                  num_anchors, 
                                  grid_y, 
                                  grid_x, 
                                  prediction_channels])

    # separate into bounding boxes, objectness score and classes
    output_bboxes = output_reshaped[:, :, :, :, :4] # first four indices are bounding box predictions
    output_objectness = output_reshaped[:, :, :, 4] # fifth index is objectness score
    output_classes = output_reshaped[:, :, :, :, 5:] # 6th onwards are class predictons
    predicted_objectness = tf.math.sigmoid(output_objectness)
    predicted_classes = tf.math.sigmoid(output_classes)

    # get predictions
    # first get anchor offsets to be added
    w_index = 0
    h_index = 1
    w_offsets, h_offsets = get_anchor_wh_offsets(w_index,
                                                 h_index,
                                                 anchors,
                                                 grid_x,
                                                 grid_y,
                                                 batch_size)

    predicted_x = (tf.math.sigmoid(output_bboxes[..., 0]) + center_x) / grid_x
    predicted_y = (tf.math.sigmoid(output_bboxes[..., 1]) + center_y) / grid_y
    predicted_w = (tf.exp(output_bboxes[..., 2]) * w_offsets)/image_width
    predicted_h = (tf.exp(output_bboxes[..., 3]) * h_offsets)/image_height

    # get iou between groundtruth and prior box anchors
    iou_anchor_gt = box_iou_tf(anchors_normalized_bbox, ground_truth_wh_bbox)
    max_iou_anchor_indices = tf.cast(tf.argmax(iou_anchor_gt, axis=-1), dtype=INT)
    max_iou_anchor_indices_extra_dim = tf.stack([tf.zeros_like(max_iou_anchor_indices, dtype=INT),
                                                 max_iou_anchor_indices],
                                                axis=-1)

    # ground truth prior box indiices
    priors_x = tf.cast(ground_truth_x_scaled, dtype=INT)
    priors_y = tf.cast(ground_truth_y_scaled, dtype=INT)
    prior_indices = tf.stack([ground_truth_batch_indices,
                              max_iou_anchor_indices,
                              priors_y,
                              priors_x])
    anchors_broadcasted = tf.broadcast_to(anchors_tensor,
                                          (num_gt, ) + anchors.shape)
    matched_anchors = tf.gather_nd(anchors_broadcasted, max_iou_anchor_indices_extra_dim)
    # get the converted gt and the matched priors from predictions
    priors_from_preds = tf.gather_nd(output_bboxes, prior_indices)
    converted_gt = convert_gt_for_loss(ground_truth_bbox,
                                       priors_x, priors_y,
                                       matched_anchors,
                                       image_width,
                                       image_height,
                                       grid_x,
                                       grid_y)
    # at this step, you can get localization loss from priors_from_preds
    # and converted_gt
    localization_loss = get_localization_loss(converted_gt, priors_from_preds)

    # get no_obj loss, obj loss and classification loss.
    # noobj loss
    predicted_bboxes = tf.stack([predicted_x,
                                 predicted_y,
                                 predicted_w,
                                 predicted_h],
                                axis=-1)
    predicted_bboxes_flattened = tf.reshape(predicted_bboxes, [-1, 4]) # flatten all bounding boxes to a Tensor of rank 2
    iou_predictions_gt = box_iou_tf(predicted_bboxes_flattened, ground_truth_bbox)
 
    # get noobj_masks
    noobj_mask_candidates = tf.where(
        tf.math.reduce_max(iou_predictions_gt,
                           axis=-1) > 0.5, # ignore_threshold
        False,
        True
    )
    noobj_mask = tf.cast(noobj_mask_candidates, dtype=INT)
    noobj_gt = tf.boolean_mask(tf.cast(noobj_mask, dtype=FLOAT),
                               noobj_mask)
    noobj_predictions = tf.boolean_mask(tf.reshape(predicted_objectness, [-1]))
    # get BCE Loss for no objectness
    noobj_loss = get_noobj_loss(noobj_gt, noobj_predictions)

    # Objectness
    objectness_predictions = tf.gather_nd(predicted_objectness,
                                          prior_indices)
    objectness_gt = tf.ones_like(objectness_predictions)
    # get BCE Loss for objectness
    obj_loss = get_obj_loss(objectness_gt, objectness_predictions)

    # bce loss for classes
    class_predictions = tf.gather_nd(predicted_classes, prior_indices)
    class_gt = ground_truth_classes_onehot
    # get bce loss for classes
    class_loss = get_class_loss(class_gt, class_predictions)

    # final losses per detection grid
    losses['regression_loss'] = localization_loss * 10.0 # scale
    losses['objectness_loss'] = obj_loss * 10.0 # scale
    losses['noobjectness_loss'] = noobj_loss * 10.0 # scale
    losses['classification_loss'] = class_loss * 10.0 # scale
    losses['total_loss'] = localization_loss + obj_loss + noobj_loss + class_loss

    return losses


def detection_loss(detection_outputs,
                   ground_truth,
                   batch_size,
                   image_height,
                   image_width,
                   num_classes,
                   anchors_list):
    """detection_loss.

    :param detection_outputs:
    :param ground_truth:
    :param batch_size:
    :param image_height:
    :param image_width:
    :param num_classes:
    :param anchors_list:
    """
    print("In Loss")
    # check detection output is list
    assert type(detection_outputs) == list, "detection_outputs" \
        "type {} is not of type list".format(type(detection_outputs))
    # check size of list of anchors is equal to length of detection outputs
    # i.e. 3 anchors and 3 detection_outputs.
    assert len(detection_outputs) == len(anchors_list), \
        "Number of outputs from detection_outputs({})" \
        "is not equal to anchors_list({})".format(len(detection_outputs),
                                                  len(anchors_list))
    # check each detection output tensor is a rank 4 tensor
    # [batch_index, grid_x, grid_y, num_classes + 5]
    for detection_output in detection_outputs:
        assert len(detection_output.shape) == 4, \
            "detection_outputs member not in shape."\
            "Has shape {}".format(detection_output.shape)

    # get some required tensors
    # grid_shapes
    grid_shapes = get_grid_shapes(detection_outputs)

    detection_losses = []
    for detection_index, detection_output in enumerate(detection_outputs):
        current_detection_loss = get_loss_per_grid_detection(detection_output,
                                                             ground_truth,
                                                             anchors_list[detection_index],
                                                             grid_shapes,
                                                             (image_width, image_height),
                                                             batch_size,
                                                             num_classes)
        detection_losses.append(current_detection_loss)

    return sum([loss['total_loss'] for loss in detection_losses])
