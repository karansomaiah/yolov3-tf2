#!/usr/bin/env python

import tensorflow as tf
import tensorflow.keras as keras

"""
Set detection losses for outputs
"""


def calculate_loss()


def wh_iou(tensora_a, tensor_b):
    """
    ground_truth x anchor
    """
    a_shape = tensor_a.shape
    b_shape = tensor_b.shape

    combined = tf.concat(
        [
            tf.broadcast_to(
                tf.reshape(
                    a,
                    shape=(a_shape[0], 1, a_shape[1], 1)
                ),
                [a_shape[0], b_shape[0], a_shape[1], 1]
            ),
            tf.broadcast_to(
                tf.reshape(
                    b,
                    shape=(1, b_shape[0], b_shape[1], 1)
                ),
                [a_shape[0], b_shape[0], b_shape[1], 1]
            )
        ],
        axis=-1
    )

    inter_area = tf.reduce_prod(tf.reduce_min(combined, axis=-1), axis=-1)
    outer_area = tf.reduce_sum(tf.reduce_prod(combined, axis=-2), axis=-1) - inter_area
    iou = outer_area/inter_area
    return iou


def generate_gt(pred_boxes,
                pred_cls,
                target,
                anchors,
                ignore_thresh):
    """ 
    Build groud truth and create masks to calculate loss against    

        Assumption here is,
        target is of style
        batch_index, class, x, y, w, h
    """
    BATCH_SIZE = pred_boxes.shape(0)
    ANCHOR_SIZE = pred_boxes.shape(1)
    GRID_X, GRID_Y = pred_boxes.shape(2), pred_boxes.shape(3)
    CLASSES_SIZE = pred_cls.shape(-1)

    GRID_SHAPE = (BATCH_SIZE, ANCHOR_SIZE, GRID_X, GRID_Y)

    obj_mask = tf.zeros(GRID_SHAPE, dtype=tf.float32)
    noobj_mask = tf.ones(GRID_SHAPE, dtype=tf.float32)  # odd man out.
    class_mask = tf.zeros(GRID_SHAPE, dtype=tf.float32)
    iou_scores = tf.zeros(GRID_SHAPE, dtype=tf.float32)
    tx = tf.zeros(GRID_SHAPE, dtype=tf.float32)
    ty = tf.zeros(GRID_SHAPE, dtype=tf.float32)
    tw = tf.zeros(GRID_SHAPE, dtype=tf.float32)
    th = tf.zeros(GRID_SHAPE, dtype=tf.float32)

    target_bboxes = target[:, 2:6] * GRID_X

    gxy = target_bboxes[:, :2]
    gwh = target_bboxes[:, 2:]

    ious = wh_iou(gwh, tf.convert_to_tensor(anchors))



def get_grid_count(tensor):
    return tensor.shape(1), tensor.shape(2)


def get_grid_offsets(grid_x_num, grid_y_num, num_anchors):
    basic_cx = tf.broadcast_to(tf.range(grid_x_num))
    basic_cy = tf.transpose(cx)
    cx = tf.broadcast_to(cx, [1, num_anchors, grid])
    return cx, cy


def get_scaled_anchors(anchors, feature_map_shape, image_shape):
    stride_w = int(image_shape[0]/feature_map_shape[0])
    stride_h = int(image_shape[1]/feature_map_shape[1])

    scaled_anchors = [((anchor_w/stride_w), (anchor_h/stride_h))
                      for anchor_w, anchor_h in anchors]

    scaled_anchor_w = tf.convert_to_tensor(
        [w for w, _ in scaled_anchors], dtype=tf.float32)

    scaled_anchor_w_reshaped = tf.reshape(
        scaled_anchor_w, [-1, len(anchors), 1, 1])

    scaled_anchor_h = tf.convert_to_tensor(
        [h for _, h in scaled_anchors], dtype=tf.float32)

    scaled_anchor_h_reshaped = tf.reshape(
        scaled_anchor_h, [-1, len(anchors), 1, 1])
    return tf.convert_to_tensor([[anchor_w, anchor_h] for anchor_w, anchor_h in scaled_anchors], dtype=tf.float32), scaled_anchor_w_reshaped, scaled_anchor_h_reshaped


def stage_outputs(detection_tensor):
    x = tf.sigmoid(detection_output[..., 0])
    y = tf.sigmoid(detection_output[..., 1])
    w = tf.exp(detection_output[..., 2])
    h = tf.exp(detection_output[..., 3])

    pred_conf = tf.sigmoid(detection_output[..., 4])
    pred_cls = tf.sigmoid(detection_output[..., 5:])

    return x, y, w, h, pred_conf, pred_cls


def segregate_output(detection_output,
                     targets,
                     anchors,
                     im_height,
                     im_width):
    """
    Segragate output separates the detection output in tx, ty, tw, th, confidences for each class.
    Assume the detection_output tensor is of shape [batch_size, grid_x, grid_y, (num_classes+5) * 3]

    detection_output is a list of outputs from each of the yolo layers
    """
    BATCH_SIZE = detection_output[0].shape(0)
    num_anchors = len(anchors)

    # Determine grid sizes
    feature_map_shapes = [(get_grid_count(output))
                          for output in detection_output]
    feature_map_offsets = [(get_grid_offsets(grid_x, grid_y))
                           for grid_x, grid_y in feature_map_shapes]
    feature_map_grid_x_offsets = [tf.reshape(
        grid_offset[0]) for grid_offset in feature_map_offsets]
    feature_map_grid_y_offsets = [tf.reshape(
        grid_offset[1]) for grid_offset in feature_map_offsets]

    for detection_index, detection in enumerate(detection_output):
        # reshape to [BATCH_SIZE, NUM_ANCHORS, GRID_X, GRID_Y, NUM_CLASSES+5]
        detection_reshaped = tf.reshape(detection, [
                                        BATCH_SIZE,
                                        num_anchors,
                                        feature_map_shapes[detection_index][0],
                                        feature_map_shapes[detection_index][1],
                                        -1])
        detection_anchors, detection_anchor_w, detection_anchor_h = get_scaled_anchors(anchors[detection_index],
                                                                                       (feature_map_shapes[detection_index][0], feature_map_shapes[detection_index][1]),
                                                                                       (im_width, im_height))

        x, y, w, h, pred_conf, pred_cls = stage_outputs(detection)

        x = x + feature_map_grid_x_offsets[detection_index]
        y = y + feature_map_grid_y_offsets[detection_index]
        w = w + detection_anchor_w
        h = h + detection_anchor_h

        predicted_bboxes = tf.stack([x, y, w, h])

        outputs = None  # To-Do

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = generate_gt(
            pred_boxes=predicted_bboxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=detection_anchors,
            ignore_threshold=0.6)
