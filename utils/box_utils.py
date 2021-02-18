#!/usr/bin/env python
"""
Utility script for processing of bounding boxes
"""

import tensorflow as tf


def overlap_tf(x1, w1, x2, w2):
    """overlap_tf.

    :param x1:
    :param w1:
    :param x2:
    :param w2:
    """
    num_1 = x1.shape[0]
    num_2 = x2.shape[0]
    x1 = tf.broadcast_to(x1, [num_2, num_1])
    w1 = tf.broadcast_to(w1, [num_2, num_1])
    x2 = tf.transpose(tf.broadcast_to(x2, [num_1, num_2]))
    w2 = tf.transpose(tf.broadcast_to(w2, [num_1, num_2]))

    l1 = x1 - (w1 / 2)
    l2 = x2 - (w2 / 2)
    left = tf.math.maximum(l1, l2)
    r1 = x1 + (w1 / 2)
    r2 = x2 + (w2 / 2)
    right = tf.math.minimum(r1, r2)
    return right - left


def box_intersection_tf(box_a, box_b):
    """box_intersection_tf.

    :param box_a:
    :param box_b:
    """
    w = overlap_tf(box_a[:, 0], box_a[:, 2], box_b[:, 0], box_b[:, 2])
    w = tf.where(tf.math.greater(w, 0.0), w, 0.0)
    h = overlap_tf(box_a[:, 1], box_a[:, 3], box_b[:, 1], box_b[:, 3])
    h = tf.where(tf.math.greater(h, 0.0), h, 0.0)
    area = w * h
    return area


def box_union_tf(box_a, box_b):
    """box_union_tf.

    :param box_a:
    :param box_b:
    """
    box_a_num = box_a.shape[0]
    box_b_num = box_b.shape[0]
    intersection = box_intersection_tf(box_a, box_b)
    box_a_w = box_a[:, 2]
    box_a_h = box_a[:, 3]
    box_b_w = box_b[:, 2]
    box_b_h = box_b[:, 3]
    box_a_w = tf.broadcast_to(box_a_w, [box_b_num, box_a_num])
    box_a_h = tf.broadcast_to(box_a_h, [box_b_num, box_a_num])
    box_b_w = tf.transpose(tf.broadcast_to(box_b_w, [box_a_num, box_b_num]))
    box_b_h = tf.transpose(tf.broadcast_to(box_b_h, [box_a_num, box_b_num]))
    union = (box_a_w * box_a_h) + (box_b_w * box_b_h) - intersection
    return union


def box_iou_tf(box_a, box_b):
    """box_iou_tf.

    :param box_a:
    :param box_b:
    """
    return box_intersection_tf(box_a, box_b) / box_union_tf(box_a, box_b)
