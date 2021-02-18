import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from operator import attrgetter


class BBoxAttrObject(object):
    TP = 0
    FP = 0
    confidence = 0.0
    class_ = 0

    def __init__(self, TP, FP, confidence, class_label):
        self.TP = TP
        self.FP = FP
        self.confidence = confidence
        self.class_ = class_label


class Evaluate:
    def __init__(self, labels_list, ious=[0.5]):

        self.labels_list = labels_list
        self.iou_thresholds = ious

        self.len_gts = 0
        self.len_dts = 0

        self.dets = {labels_list[i]: [] for i in range(len(labels_list))}
        self.GT = {labels_list[i]: 0 for i in range(len(labels_list))}

        self.Pr = {labels_list[i]: [] for i in range(len(labels_list))}
        self.Re = {labels_list[i]: [] for i in range(len(labels_list))}

        self.class_wise_APs = {labels_list[i]: 0.0 for i in range(len(labels_list))}
        self.mAP = 0.0

    """
        pred_box : the coordinate for predict bounding box
        gt_box :   the coordinate for ground truth bounding box
        return :   the iou score
        the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
        the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """

    def get_iou(self, pred_box, gt_box):
        # get the coordinate of intersection
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        ih = np.maximum(iymax - iymin + 1.0, 0.0)

        # calculate the area of intersection
        inters = iw * ih

        # calculate the area of union
        uni = (
            (pred_box[2] - pred_box[0] + 1.0) * (pred_box[3] - pred_box[1] + 1.0)
            + (gt_box[2] - gt_box[0] + 1.0) * (gt_box[3] - gt_box[1] + 1.0)
            - inters
        )

        # calculate the overlaps between pred_box and gt_box
        iou = inters / uni

        return iou

    """
        inputs: per image
            gt_boxes: ground truth labels for the image - numpy array [ [x1,y1,x2,y2,class_id], [x1,y1,x2,y2,class_id], ... ]
            pred_boxes: all the predicted boxex - numpy array [[x1,y1,x2,y2],[x1,y1,x2,y2] ... ]
            scores: score for all predicted boxes [s0, s1, ...]
            pred_classes: list of corresponding prediceted classes - [class0, class1, class2 ... classN]
    """

    def consider_image(self, gt_boxes, pred_boxes, pred_scores, pred_classes):

        self.len_dts += len(pred_boxes)
        self.len_gts += len(gt_boxes)

        for i, class_ in enumerate(self.labels_list):
            gt_boxes_i = [box[0:4] for box in gt_boxes if (box[4] == i)]
            self.GT[class_] += len(gt_boxes_i)

            pred_boxes_i = [
                (pred_boxes[k], pred_scores[k])
                for k in range(len(pred_boxes))
                if (pred_classes[k] == i)
            ]
            matched_gt = [False] * len(gt_boxes_i)

            for pb in range(len(pred_boxes_i)):
                min_iou = 2
                matched_gt_index = 0
                if len(matched_gt) > 0:
                    for gtb in range(len(gt_boxes_i)):
                        if False == matched_gt[gtb]:
                            iou = self.get_iou(pred_boxes_i[pb][0], gt_boxes_i[gtb])
                            if (iou < min_iou) and (iou > self.iou_thresholds[0]):
                                min_iou = iou
                                matched_gt_index = gtb
                    if matched_gt[matched_gt_index] == False:
                        matched_gt[matched_gt_index] = True
                        bbox = BBoxAttrObject(1, 0, pred_boxes_i[pb][1], i)
                        self.dets[class_].append(bbox)
                    else:
                        bbox = BBoxAttrObject(0, 1, pred_boxes_i[pb][1], i)
                        self.dets[class_].append(bbox)
                else:
                    bbox = BBoxAttrObject(0, 1, pred_boxes_i[pb][1], i)
                    self.dets[class_].append(bbox)

    def consider_batch(
        self, batch_gt_boxes, batch_pred_boxes, batch_pred_scores, batch_pred_classes
    ):

        batch_size = len(batch_pred_scores)

        for i in range(batch_size):
            self.consider_image(
                gt_boxes=batch_gt_boxes[i],
                pred_boxes=batch_pred_boxes[i],
                pred_scores=batch_pred_scores[i],
                pred_classes=batch_pred_classes[i],
            )

    def calculate_class_wise_APs(self):

        for idx, class_ in enumerate(self.labels_list):
            self.dets[class_].sort(key=attrgetter("confidence"), reverse=True)
            tempTP = [bbox.TP for bbox in self.dets[class_]]
            tempFP = [bbox.FP for bbox in self.dets[class_]]

            tempTP = np.cumsum(tempTP)
            tempFP = np.cumsum(tempFP)

            assert len(tempTP) == len(tempFP)

            self.Pr[class_] = np.divide(tempTP, np.add(tempTP, tempFP)).tolist()
            self.Re[class_] = (tempTP / self.GT[class_]).tolist()

            ap, mpre, mrec, _ = self.calculate_average_precision(
                self.Re[class_], self.Pr[class_]
            )

            self.class_wise_APs[class_] = ap
            self.Pr[class_] = mpre
            self.Re[class_] = mrec

    def get_mAPs(self):
        self.calculate_class_wise_APs()
        self.mAP = 0.0
        for idx, class_ in enumerate(self.labels_list):
            self.mAP += self.class_wise_APs[class_]

        self.mAP = self.mAP / len(self.labels_list)

        return self.mAP, self.class_wise_APs

    # Based on Rafael Padilla's repo: https://github.com/rafaelpadilla/Object-Detection-Metrics
    def calculate_average_precision(self, rec, prec):

        mrec = rec
        mrec.insert(0, 0.0)
        mrec.append(1.0)

        mpre = prec
        mpre.insert(0, 0.0)
        mpre.append(0.0)

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        return [ap, mpre[0 : len(mpre) - 1], mrec[0 : len(mpre) - 1], ii]


# for debugging
# if __name__=='__main__':
#
#    gt_boxes = [[
#                [ 282.0133,  348.3891,  294.9497,  380.0345,    0.0000],
#                [1213.0696,   95.6777, 1248.7721,  186.5343,    0.0000],
#                # [1173.2874,  173.3222, 1201.8337,  245.1273,    0.0000],
#                [ 456.3774,  435.4330,  467.3437,  443.9970,    0.0000],
#                [ 110.1237,  304.7455,  123.7689,  338.7431,    2.0000],
#                [ 184.0777,  322.9569,  197.5335,  352.6584,    2.0000],],]
#
#    pred_boxes = [[
#                [ 282.0133,  348.3891,  294.9497,  380.0345],
#                [1213.0696,   95.6777, 1248.7721,  186.5343],
#                [1173.2874,  173.3222, 1201.8337,  245.1273],
#                # [ 456.3774,  435.4330,  467.3437,  443.9970],
#                [ 110.1237,  304.7455,  123.7689,  338.7431],
#                [ 184.0777,  322.9569,  197.5335,  352.6584],],]
#
#    pred_labels = [[0,0,0,2,2]]
#
#    pred_scores = [[0.95,0.7,0.66,0.9,0.95]]
#
#    labels_list = [ 'TrafficLight-NotVisible', 'TrafficLight-Off', 'TrafficLight-Red',
#                'TrafficLight-RedLeft', 'TrafficLight-RedRight', 'TrafficLight-RedStraight',
#                'TrafficLight-Yellow', 'TrafficLight-Green', 'TrafficLight-GreenLeft', 'TrafficLight-GreenRight',
#                'TrafficLight-GreenStraight', 'PedestrianLight-Walk', 'PedestrianLight-NoWalk' ]
#
#    #Usage
#    # create Evaluate object
#    eval = Evaluate(labels_list=labels_list, ious=[0.5])
#
#    # While batch evaluation
#    #### call consider_batch after every batch inference, and provide:
#    #### gt_boxes [ [ [x1,y1,x2,y2,class], ...], ... ]
#    #### pred_boxes [ [ [x1,y2,x2,y2], ... ], ... ]
#    #### pred_scores [ [s, ... ], ... ]
#    #### pred_labels [ [l, ... ], ... ]
#    eval.consider_batch(batch_gt_boxes=gt_boxes, batch_pred_boxes=pred_boxes,
#                            batch_pred_scores=pred_scores, batch_pred_classes=pred_labels)
#
#    ## Once all the batches are done, call eval.get_mAPs()
#    ## return mAP(float), and class_wise_APs(dict with keys as your label names)
#    mAP, class_wise_APs = eval.get_mAPs()
#
#    print(mAP)
#    print(class_wise_APs)
