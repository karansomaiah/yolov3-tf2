import numpy as np
import tensorflow as tf
from utils.box_utils import box_iou_tf
from utils.utils import INT, FLOAT, BOOL


def DetectionLoss(
    image_height,
    image_width,
    num_classes,
    num_anchors,
    anchors_list,
):
    batch_size = None
    image_height = image_height
    image_width = image_width
    num_classes = num_classes
    anchors_list = anchors_list
    current_loss_map = {
        "mse": tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        ),
        "bce": tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        ),
    }
    loss_list = []

    def get_grid_shapes(detection):
        return (detection.shape[1], detection.shape[2]), detection.shape[0]

    def get_grid_offsets(grid_x_num, grid_y_num, batch_size, anchor_size):
        """"""
        intermediate_center_x_offset = tf.broadcast_to(
            tf.range(grid_x_num, dtype=tf.float32), [grid_y_num, grid_x_num]
        )
        center_x_offsets = tf.broadcast_to(
            intermediate_center_x_offset,
            [batch_size, anchor_size, grid_y_num, grid_x_num],
        )
        intermediate_center_y_offset = tf.broadcast_to(
            tf.range(grid_y_num, dtype=tf.float32), [grid_x_num, grid_y_num]
        )
        intermediate_center_y_offset = tf.transpose(intermediate_center_y_offset)
        center_y_offsets = tf.broadcast_to(
            intermediate_center_y_offset,
            [batch_size, anchor_size, grid_y_num, grid_x_num],
        )
        return center_x_offsets, center_y_offsets

    def get_anchor_wh_offsets(w_index, h_index, anchors, grid_x, grid_y, batch_size):
        anchor_size = anchors.shape[0]
        w_offsets = tf.broadcast_to(
            tf.expand_dims(
                tf.broadcast_to(
                    tf.expand_dims(
                        tf.broadcast_to(
                            tf.expand_dims(anchors[:, w_index], axis=-1),
                            [anchor_size, grid_y],
                        ),
                        axis=-1,
                    ),
                    [anchor_size, grid_y, grid_x],
                ),
                axis=0,
            ),
            [batch_size, anchor_size, grid_y, grid_x],
        )

        h_offsets = tf.broadcast_to(
            tf.expand_dims(
                tf.broadcast_to(
                    tf.expand_dims(
                        tf.broadcast_to(
                            tf.expand_dims(anchors[:, h_index], axis=-1),
                            [anchor_size, grid_y],
                        ),
                        axis=-1,
                    ),
                    [anchor_size, grid_y, grid_x],
                ),
                axis=0,
            ),
            [batch_size, anchor_size, grid_y, grid_x],
        )
        return w_offsets, h_offsets

    def convert_gt_for_loss(
        ground_truth,
        prior_x,
        prior_y,
        matched_anchors,
        image_width,
        image_height,
        grid_x,
        grid_y,
    ):
        w_index = 0
        h_index = 1
        x_index, y_index, w_ind, h_ind = 0, 1, 2, 3

        ground_truth_x = (ground_truth[:, x_index] * grid_x) - tf.cast(
            prior_x, dtype=FLOAT
        )
        ground_truth_y = (ground_truth[:, y_index] * grid_y) - tf.cast(
            prior_y, dtype=FLOAT
        )
        ground_truth_w = tf.math.log(
            ground_truth[:, w_ind] * (image_width / matched_anchors[:, w_index])
        )
        ground_truth_h = tf.math.log(
            ground_truth[:, h_ind] * (image_height / matched_anchors[:, h_index])
        )

        return tf.stack(
            [ground_truth_x, ground_truth_y, ground_truth_w, ground_truth_h], axis=-1
        )

    def get_localization_loss(loss_map, true, pred):
        return loss_map["mse"](true, pred)

    def get_noobj_loss(loss_map, true, pred):
        return loss_map["bce"](true, pred)

    def get_obj_loss(loss_map, true, pred):
        return loss_map["bce"](true, pred)

    def get_class_loss(loss_map, true, pred):
        return loss_map["bce"](true, pred)

    def get_loss_per_grid_detection(
        detection_output,
        loss_map,
        ground_truth,
        anchors,
        num_anchors,
        grid_shape,
        image_wh,
        batch_size,
        num_classes,
    ):
        # print("Detection output: {}".format(detection_output.shape))
        # print(detection_output[0, :2, :2, :])
        losses = {
            "regression_loss": 0.0,
            "objectness_loss": 0.0,
            "noobjectness_loss": 0.0,
            "classification_loss": 0.0,
            "total_loss": 0.0,
            "valid": True,
        }
        valid_matches = True
        # num_anchors = len(anchors)
        grid_x, grid_y = grid_shape
        image_width, image_height = image_wh
        width_downsampling_factor = image_width / grid_x
        height_downsampling_factor = image_height / grid_y
        num_channels = detection_output.shape[-1]
        prediction_channels = tf.cast(num_channels / num_anchors, dtype=INT)
        num_gt = ground_truth.shape[0]

        # ground truth tensors
        ground_truth_bbox = ground_truth[:, 1:-1]
        ground_truth_wh_bbox = tf.concat(
            [tf.zeros_like(ground_truth_bbox[:, -2:]), ground_truth_bbox[:, -2:]],
            axis=-1,
        )
        ground_truth_classes = tf.cast(ground_truth[:, -1], dtype=INT)
        ground_truth_classes_onehot = tf.one_hot(
            ground_truth_classes, depth=num_classes
        )
        ground_truth_batch_indices = tf.cast(ground_truth[:, 0], dtype=INT)
        ground_truth_x = ground_truth_bbox[:, 0]
        ground_truth_y = ground_truth_bbox[:, 1]
        ground_truth_x_scaled = ground_truth_x * grid_x
        ground_truth_y_scaled = ground_truth_y * grid_y

        # anchors tensor
        anchors_tensor = anchors  # tf.constant(anchors, dtype=FLOAT)
        anchors_normalized = tf.math.divide(
            anchors_tensor,
            tf.broadcast_to(
                tf.expand_dims(
                    tf.constant([float(image_width), float(image_height)], dtype=FLOAT),
                    axis=0,
                ),
                [anchors_tensor.shape[0], anchors_tensor.shape[1]],
            ),
        )
        anchors_normalized_bbox = tf.concat(
            [tf.zeros_like(anchors_normalized), anchors_normalized], axis=-1
        )

        # x & y - offsets
        center_x, center_y = get_grid_offsets(grid_x, grid_y, batch_size, num_anchors)

        # reshape output tensor to [batch_size,
        #                           anchor_size,
        #                           grid_x,
        #                           grid_y,
        #                           pred_channels]
        output_reshaped = tf.reshape(
            detection_output,
            [batch_size, num_anchors, grid_y, grid_x, prediction_channels],
        )

        # separate into bounding boxes, objectness score and classes
        output_bboxes = output_reshaped[
            :, :, :, :, :4
        ]  # first four indices are bounding box predictions
        output_objectness = output_reshaped[
            :, :, :, :, 4
        ]  # fifth index is objectness score
        output_classes = output_reshaped[
            :, :, :, :, 5:
        ]  # 6th onwards are class predictons
        predicted_objectness = tf.math.sigmoid(output_objectness)
        predicted_classes = tf.math.sigmoid(output_classes)

        # get predictions
        # first get anchor offsets to be added
        w_index = 0
        h_index = 1
        w_offsets, h_offsets = get_anchor_wh_offsets(
            w_index, h_index, anchors_tensor, grid_x, grid_y, batch_size
        )

        predicted_x = (tf.math.sigmoid(output_bboxes[..., 0]) + center_x) / grid_x
        predicted_y = (tf.math.sigmoid(output_bboxes[..., 1]) + center_y) / grid_y
        predicted_w = (tf.exp(output_bboxes[..., 2]) * w_offsets) / image_width
        predicted_h = (tf.exp(output_bboxes[..., 3]) * h_offsets) / image_height

        # get iou between groundtruth and prior box anchors
        iou_anchor_gt = box_iou_tf(anchors_normalized_bbox, ground_truth_wh_bbox)
        if tf.reduce_any(iou_anchor_gt > 0.5):
            valid_matches = True
        else:
            valid_matches = True

        max_iou_anchor_indices = tf.cast(tf.argmax(iou_anchor_gt, axis=-1), dtype=INT)
        max_iou_anchor_indices_extra_dim = tf.stack(
            [tf.zeros_like(max_iou_anchor_indices, dtype=INT), max_iou_anchor_indices],
            axis=-1,
        )

        # ground truth prior box indiices
        priors_x = tf.cast(ground_truth_x_scaled, dtype=INT)
        priors_y = tf.cast(ground_truth_y_scaled, dtype=INT)
        prior_indices = tf.stack(
            [ground_truth_batch_indices, max_iou_anchor_indices, priors_y, priors_x],
            axis=-1,
        )

        anchors_broadcasted = tf.broadcast_to(
            anchors_tensor, (num_gt,) + anchors_tensor.shape
        )
        matched_anchors = tf.gather_nd(
            anchors_broadcasted, max_iou_anchor_indices_extra_dim
        )
        # get the converted gt and the matched priors from predictions
        priors_from_preds = tf.gather_nd(output_bboxes, prior_indices)
        converted_gt = convert_gt_for_loss(
            ground_truth_bbox,
            priors_x,
            priors_y,
            matched_anchors,
            image_width,
            image_height,
            grid_x,
            grid_y,
        )

        # at this step, you can get localization loss from priors_from_preds
        # and converted_gt
        localization_loss = get_localization_loss(
            loss_map, converted_gt, priors_from_preds
        )

        # get no_obj loss, obj loss and classification loss.
        # noobj loss
        predicted_bboxes = tf.stack(
            [predicted_x, predicted_y, predicted_w, predicted_h], axis=-1
        )
        predicted_bboxes_flattened = tf.reshape(
            predicted_bboxes, [-1, 4]
        )  # flatten all bounding boxes to a Tensor of rank 2

        candidate_ignore_threshold = tf.where(iou_anchor_gt > 0.5)
        candidate_it_gt_indices = candidate_ignore_threshold[:, 0]
        candidate_it_anchor_indices = candidate_ignore_threshold[:, 1]

        # create positive objectness mask
        batch_index_for_candidates = tf.cast(
            tf.gather_nd(
                ground_truth_batch_indices,
                tf.expand_dims(candidate_it_gt_indices, axis=-1),
            ),
            dtype=INT,
        )
        anchor_index_for_candidates = tf.cast(candidate_it_anchor_indices, dtype=INT)
        y_index_for_candidates = tf.cast(
            tf.gather_nd(priors_y, tf.expand_dims(candidate_it_gt_indices, axis=-1)),
            dtype=INT,
        )
        x_index_for_candidates = tf.cast(
            tf.gather_nd(priors_x, tf.expand_dims(candidate_it_gt_indices, axis=-1)),
            dtype=INT,
        )
        positive_objects_indices = tf.stack(
            [
                batch_index_for_candidates,
                anchor_index_for_candidates,
                y_index_for_candidates,
                x_index_for_candidates,
            ],
            axis=-1,
        )
        positive_objects_indices = tf.concat(
            [prior_indices, positive_objects_indices], axis=0
        )

        # create objectness mask
        objectness_gt_values = tf.cast(
            tf.ones_like(positive_objects_indices[:, 0]), dtype=INT
        )
        objectness_mask = tf.scatter_nd(
            positive_objects_indices,
            objectness_gt_values,
            [batch_size, num_anchors, grid_y, grid_x],
        )
        objectness_mask = tf.where(objectness_mask != 0, 1.0, 0.0)
        objectness_mask = tf.cast(objectness_mask, INT)
        noobjectness_mask = tf.cast(
            tf.logical_not(tf.cast(objectness_mask, dtype=tf.bool)), INT
        )
        objectness_gt_everything = tf.cast(objectness_mask, dtype=FLOAT)
        # get objectness loss
        objects_positives_gt = tf.boolean_mask(
            objectness_gt_everything, objectness_mask
        )
        objects_positives_preds = tf.boolean_mask(predicted_objectness, objectness_mask)
        objects_negative_gt = tf.boolean_mask(
            objectness_gt_everything, noobjectness_mask
        )
        objects_negative_preds = tf.boolean_mask(
            predicted_objectness, noobjectness_mask
        )
        obj_loss = get_noobj_loss(
            loss_map, objects_positives_gt, objects_positives_preds
        )
        noobj_loss = get_obj_loss(loss_map, objects_negative_gt, objects_negative_preds)

        # bce loss for classes
        class_predictions = tf.gather_nd(predicted_classes, prior_indices)
        class_gt = ground_truth_classes_onehot

        # get bce loss for classes
        class_loss = get_class_loss(loss_map, class_gt, class_predictions)

        # final losses per detection grid
        losses["regression_loss"] = localization_loss * 1.0  # scale
        losses["objectness_loss"] = obj_loss * 1.0  # scale
        losses["noobjectness_loss"] = noobj_loss * 100.0  # scale
        losses["classification_loss"] = class_loss * 1.0  # scale
        losses["total_loss"] = (
            losses["regression_loss"]
            + losses["objectness_loss"]
            + losses["noobjectness_loss"]
            + losses["classification_loss"]
        )

        losses["valid"] = valid_matches
        return losses

    def call(y_true, y_pred):
        # i.e y_true = ground_truth
        #     y_pred = detection_outputs
        ground_truth = y_true
        detection_outputs = y_pred
        grid_shape, batch_size = get_grid_shapes(detection_outputs)

        # for detection_index, detection_output in enumerate(detection_outputs):
        #    current_detection_loss = get_loss_per_grid_detection(
        #        detection_output,
        #        current_loss_map,
        #        ground_truth,
        #        anchors_list[detection_index],
        #        grid_shapes[detection_index],
        #        (image_width, image_height),
        #        batch_size,
        #        num_classes,
        #    )
        #    detection_losses.append(current_detection_loss)
        loss = get_loss_per_grid_detection(
            detection_outputs,
            current_loss_map,
            ground_truth,
            anchors_list,
            num_anchors,
            grid_shape,
            (image_width, image_height),
            batch_size,
            num_classes,
        )

        return loss["total_loss"]
        # return loss

    return call
