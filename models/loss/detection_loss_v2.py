#!/usr/bin/env python
"""
"""
import numpy as np
import tensorflow as tf

#from utils.box_utils import box_iou_tf
#from utils.utils import INT, FLOAT, BOOL


def overlap_tf(x1, w1, x2, w2):
    num_1 = x1.shape[0]
    num_2 = x2.shape[0]
    x1 = tf.broadcast_to(x1, [num_2, num_1])
    w1 = tf.broadcast_to(w1, [num_2, num_1])
    x2 = tf.transpose(tf.broadcast_to(x2, [num_1, num_2]))
    w2 = tf.transpose(tf.broadcast_to(w2, [num_1, num_2]))

    l1 = x1 - (w1 / 2)
    l2 = x2 - (w2 / 2)
    l = tf.math.maximum(l1, l2)
    r1 = x1 + (w1 / 2)
    r2 = x2 + (w2 / 2)
    r = tf.math.minimum(r1, r2)
    return r - l


def box_intersection_tf(a, b):
    w = overlap_tf(a[:, 0], a[:, 2], b[:, 0], b[:, 2])
    w = tf.where(tf.math.greater(w, 0.0), w, 0.0)
    h = overlap_tf(a[:, 1], a[:, 3], b[:, 1], b[:, 3])
    h = tf.where(tf.math.greater(h, 0.0), h, 0.0)
    area = w * h
    return area


def box_union_tf(a, b):
    a_num = a.shape[0]
    b_num = b.shape[0]
    intersection = box_intersection_tf(a, b)
    a_w = a[:, 2]
    a_h = a[:, 3]
    b_w = b[:, 2]
    b_h = b[:, 3]
    a_w = tf.broadcast_to(a_w, [b_num, a_num])
    a_h = tf.broadcast_to(a_h, [b_num, a_num])
    b_w = tf.transpose(tf.broadcast_to(b_w, [a_num, b_num]))
    b_h = tf.transpose(tf.broadcast_to(b_h, [a_num, b_num]))
    union = (a_w * a_h) + (b_w * b_h) - intersection
    return union


def box_iou_tf(a, b):
    return box_intersection_tf(a, b) / box_union_tf(a, b)


def getLoss(
    batch_size, anchors_list, num_anchors, image_width, image_height, top_k=100
):
    """"""
    # converting list of anchors into tensor & recording the number of anchors for the current loss
    anchor_tensor = tf.constant(anchors_list, dtype=tf.float32)
    num_anchors = anchor_tensor.shape[0]

    def loss(y_true, y_pred):

        input_dims = tf.cast(
            tf.broadcast_to([[image_width, image_height]], tf.shape(anchor_tensor)),
            dtype=tf.float32,
        )
        anchors_normalized = anchor_tensor / input_dims
        is_gt = tf.where(y_true[:, :, -1] == 0.0, False, True)
        batch_size = y_pred.shape[0]
        grid_y = y_pred.shape[1]
        grid_x = y_pred.shape[2]
        num_channels = y_pred.shape[3]
        num_prediction_channels = tf.cast(num_channels / num_anchors, dtype=tf.int32)
        broadcasted_anchors = tf.broadcast_to(
            input=anchor_tensor, shape=[batch_size, top_k, num_anchors, 2]
        )
        broadcasted_batch = tf.broadcast_to(
            input=tf.expand_dims(tf.range(batch_size), axis=1),
            shape=[batch_size, top_k],
        )
        y_pred_reshaped = tf.reshape(
            y_pred, [batch_size, num_anchors, grid_y, grid_x, num_prediction_channels]
        )
        predicted_bboxes = y_pred_reshaped[..., :4]
        predicted_objectness = tf.math.sigmoid(y_pred_reshaped[..., 4])
        predicted_classes = tf.math.sigmoid(y_pred_reshaped[..., 5:])

        # get anchor wh and ground truth wh iou
        anchors_wh = tf.concat(
            values=[
                tf.zeros_like(anchors_normalized[:, :2], dtype=tf.float32),
                anchors_normalized[:, :2],
            ],
            axis=-1,
        )
        gt_wh = tf.concat(
            values=[tf.zeros_like(y_true[:, :, -3:-1]), y_true[:, :, -3:-1]], axis=-1
        )
        anchors_wh = tf.reshape(anchors_wh, [-1, 4])  # flatten to store num_anchors, 4
        gt_wh = tf.reshape(gt_wh, [-1, 4])  # flatten to store batch_size * 100, 4

        # @To-Do
        # create center_offsets
        intermediate_center_x_offset = tf.broadcast_to(
            input=tf.range(grid_x, dtype=tf.float32), shape=[grid_y, grid_x]
        )
        center_x_offsets = tf.broadcast_to(
            input=intermediate_center_x_offset,
            shape=[batch_size, num_anchors, grid_y, grid_x],
        )
        intermediate_center_y_offset = tf.broadcast_to(
            input=tf.range(grid_y, dtype=tf.float32), shape=[grid_x, grid_y]
        )
        center_y_offsets = tf.broadcast_to(
            input=intermediate_center_y_offset,
            shape=[batch_size, num_anchors, grid_y, grid_x],
        )

        # get iou between anchors and ground truths
        # get the maximum per anchor
        iou = box_iou_tf(anchors_wh, gt_wh)
        iou_reshape = tf.reshape(
            tensor=iou,
            shape=[batch_size, top_k, num_anchors],
        )
        iou_argmax_per_anchor = tf.argmax(iou_reshape, axis=-1, output_type=tf.int32)

        # ground truth bounding box
        gt_bb = y_true[:, :, :-1]  # only x, y, w, h
        prior_x = tf.cast(
            gt_bb[:, :, 0] * tf.cast(grid_x, dtype=tf.float32), dtype=tf.int32
        )
        prior_y = tf.cast(
            gt_bb[:, :, 1] * tf.cast(grid_x, dtype=tf.float32), dtype=tf.int32
        )
        selected_anchors_per_gt = tf.stack(
            values=[
                broadcasted_batch,
                tf.broadcast_to(tf.range(top_k), shape=[batch_size, top_k]),
                iou_argmax_per_anchor,
            ],
            axis=-1,
        )

        best_anchors_per_gt = tf.gather_nd(
            params=broadcasted_anchors, indices=selected_anchors_per_gt
        )
        indices_to_subset = tf.stack(
            values=[
                broadcasted_batch,
                iou_argmax_per_anchor,
                prior_y,
                prior_x,
            ],
            axis=-1,
        )
        gt_indices = tf.boolean_mask(tensor=indices_to_subset, mask=is_gt)

        # @To-Do Clean up offset creation.
        # get the predicted bboxes and calculate MSE LOSS
        w_index = 0
        h_index = 1
        w_offsets = tf.broadcast_to(
            tf.expand_dims(
                tf.broadcast_to(
                    tf.expand_dims(
                        tf.broadcast_to(
                            tf.expand_dims(anchor_tensor[:, w_index], axis=-1),
                            [num_anchors, grid_y],
                        ),
                        axis=-1,
                    ),
                    [num_anchors, grid_y, grid_x],
                ),
                axis=0,
            ),
            [batch_size, num_anchors, grid_y, grid_x],
        )
        h_offsets = tf.broadcast_to(
            tf.expand_dims(
                tf.broadcast_to(
                    tf.expand_dims(
                        tf.broadcast_to(
                            tf.expand_dims(anchor_tensor[:, h_index], axis=-1),
                            [num_anchors, grid_y],
                        ),
                        axis=-1,
                    ),
                    [num_anchors, grid_y, grid_x],
                ),
                axis=0,
            ),
            [batch_size, num_anchors, grid_y, grid_x],
        )

        y_pred_bb = tf.gather_nd(params=predicted_bboxes, indices=gt_indices)

        # gt to calculate loss against
        gt_tx = (gt_bb[..., 0] * tf.cast(grid_x, tf.float32)) - tf.cast(
            prior_x, dtype=tf.float32
        )
        gt_ty = (gt_bb[..., 1] * tf.cast(grid_y, tf.float32)) - tf.cast(
            prior_y, dtype=tf.float32
        )
        gt_tw = tf.math.log(gt_bb[..., 2] * (image_width / best_anchors_per_gt[..., 0]))
        gt_th = tf.math.log(gt_bb[..., 3] * (
            image_height / best_anchors_per_gt[..., 1]
        ))
        gt_t = tf.stack([gt_tx, gt_ty, gt_tw, gt_th], axis=-1)
        y_true_bb = tf.boolean_mask(gt_t, is_gt)
        localization_loss_weight = y_true_bb[:, -2] * y_true_bb[:, -1]

        # REGRESSION LOSS
        # @To-Do get a customized loss i.e MSE, BCE, etc. from a config file passed to
        # the loss creator
        mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )
        localization_loss = mse_loss(y_true_bb, y_pred_bb,
                                     sample_weight=localization_loss_weight)

        # Objectness Loss
        candidate_objects_indices = tf.where(iou_reshape > 0.5)
        batch_index = tf.cast(candidate_objects_indices[:, 0], dtype=tf.int32)
        topk_index = tf.cast(candidate_objects_indices[:, 1], dtype=tf.int32)
        anchor_index = tf.cast(candidate_objects_indices[:, 2], dtype=tf.int32)
        gt_per_candidate = tf.cast(candidate_objects_indices[:, :2], dtype=tf.int32)
        candidate_objects_x = tf.gather_nd(prior_x, gt_per_candidate)
        candidate_objects_y = tf.gather_nd(prior_y, gt_per_candidate)
        candidates = tf.stack(
            [batch_index, anchor_index, candidate_objects_y, candidate_objects_x],
            axis=-1,
        )
        # the above list is appeneded with the ground truth one
        # since they are objects too (you know :')).
        # And this combined list will have some duplicates which will have
        # values greater than 1 when applied with scatter_nd.
        # this is rectified to 1
        all_candidates = tf.concat([candidates, gt_indices], axis=0)
        updates_for_mask = tf.ones_like(all_candidates[:, 0], dtype=tf.int32)
        objectness_mask = tf.scatter_nd(
            indices=all_candidates,
            updates=updates_for_mask,
            shape=[batch_size, num_anchors, grid_y, grid_x],
        )
        objectness_mask = tf.where(objectness_mask != 0, 1, 0)
        objects_indices = tf.where(objectness_mask != 0)
        y_pred_objectness = tf.gather_nd(predicted_objectness, objects_indices)
        y_true_objectness = tf.boolean_mask(
            tf.cast(objectness_mask, dtype=tf.float32), objectness_mask
        )

        # OBJECTNESS LOSS
        bce = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        )
        objectness_loss = bce(y_true_objectness, y_pred_objectness)

        # No Objectness Loss
        noobj_mask = tf.cast(
            tf.logical_not(tf.cast(objectness_mask, dtype=tf.bool)), dtype=tf.int32
        )
        noobj_indices = tf.where(noobj_mask != 0)
        y_pred_no_objectness = tf.gather_nd(predicted_objectness, noobj_indices)
        y_true_no_objectness = tf.boolean_mask(
            tf.cast(objectness_mask, dtype=tf.float32), noobj_mask
        )

        noobjectness_loss = bce(y_true_no_objectness, y_pred_no_objectness)

        # Class Loss
        label_class_indices = tf.expand_dims(
            tf.cast(tf.boolean_mask(y_true[:, :, -1], is_gt), dtype=tf.int32), axis=1
        )
        pred_classes_indices = tf.concat(
            values=[gt_indices, label_class_indices], axis=-1
        )
        y_pred_classes = tf.gather_nd(predicted_classes, pred_classes_indices)
        y_true_classes = tf.ones_like(y_pred_classes)
        classification_loss = bce(y_true_classes, y_pred_classes)

        # @To-Do: Add scaling configs
        # @To-Do: Add callbacks for tensorboard logging
        localization_loss = localization_loss * 1.0
        objectness_loss = objectness_loss * 1.0
        noobjectness_loss = noobjectness_loss * 100.0
        classification_loss = classification_loss * 1.0
        
        total_loss = localization_loss + objectness_loss + noobjectness_loss + classification_loss
        
        return total_loss, localization_loss, objectness_loss, noobjectness_loss, classification_loss

    return loss
