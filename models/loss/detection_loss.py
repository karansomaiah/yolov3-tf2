#!/usr/bin/env python

"""
Set detection losses for outputs
"""

def calculate_loss()

def generate_gt(inputs,
                anchors,
                grid_x,
                grid_y):
    """
    Given inputs as 
        batch_i, bx, by, bw, bh, c -> (batch_num, x_center, y_center, w_center, h_center, class_index)
        and scaled anchors, convert these to ground truths that can be used to calculate losses.
    """
    box = inputs[:, 1:-1] * grid_x
    bx = box[:, 0]
    by = box[:, 1]
    bw = box[:, 2]
    bh = box[:, 3]
