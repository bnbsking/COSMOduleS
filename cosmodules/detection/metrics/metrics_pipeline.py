from collections import Counter
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from cosmodules.detection.metrics.confusion_matrix import DetectionConfusionMatrix
from cosmodules.utils.metrics import (
    PRCurve,
    get_refined_pr_curves,
    get_ap_list,
    get_map,
    get_wmap,
    ThresholdOptimizer,
    get_confusion_axis_norm,
    get_sorted_accuracy_index_list
)
from cosmodules.utils.metrics_postprocess import deserialize


class DetectionMetricsPipeline:
    @staticmethod
    def get_gt_class_cnts(num_classes: int, labels: List[np.ndarray]) -> List[int]:
        gt_class_cnts = [0] * (num_classes - 1)
        for labels_per_img in labels:
            for label in labels_per_img:
                cid, _, _, _, _ = label
                gt_class_cnts[cid - 1] += 1
        return gt_class_cnts

    @staticmethod
    def get_pr_curves(
            num_classes: int,
            labels: List[np.ndarray],
            predictions: List[np.ndarray],
            k: int = 101
        ) -> List[PRCurve]:
        pr_curves = [
            PRCurve(
                precision=[0.] * k,
                recall=[0.] * k,
            ) for _ in range(num_classes - 1)
        ]

        for i, threshold in tqdm(enumerate(np.linspace(0, 1, k))):
            # get confusion of the threshold
            confusion = np.zeros(
                (num_classes, num_classes)
            )  # (i, j) = (gt, pd)

            for labels_per_img, predictions_per_img in zip(labels, predictions):
                img_confusion = DetectionConfusionMatrix(
                    num_classes,
                    CONF_THRESHOLD = threshold,
                    IOU_THRESHOLD = 0.5
                )
                img_confusion.process_batch(predictions_per_img, labels_per_img)
                confusion += img_confusion.get_confusion()
            
            # update pr curve at the threshold from confusion
            row_sum = confusion.sum(axis=1)
            col_sum = confusion.sum(axis=0)
            for cid in range(1, num_classes):
                pr_curves[cid-1].precision[i] = confusion[cid][cid] / col_sum[cid] if col_sum[cid] else 0
                pr_curves[cid-1].recall[i] = confusion[cid][cid] / row_sum[cid] if row_sum[cid] else 0

        return pr_curves

    @staticmethod
    def get_confusion(
            num_classes: int,
            labels: List[np.ndarray],
            predictions: List[np.ndarray],
            threshold: float = 0.5
        ) -> np.ndarray:  # int, shape=(num_classes, num_classes)
        confusion = np.zeros((num_classes, num_classes))  # row: gt, col: pd
        for labels_per_img, predictions_per_img in zip(labels, predictions):
            cm = DetectionConfusionMatrix(
                num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5
            )
            cm.process_batch(predictions_per_img, labels_per_img)
            confusion += cm.get_confusion()
        return confusion

    @staticmethod
    def get_confusion_with_img_indices(
            num_classes: int,
            labels: List[np.ndarray],
            predictions: List[np.ndarray],
            threshold: float = 0.5
        ) -> List[List[Counter[int, int]]]:  # shape=(num_classes, num_classes), each grid: Counters (img_idx -> cnts)
        confusion_with_img_indices = [
            [Counter() for _ in range(num_classes)] for _ in range(num_classes)
        ]
        for img_idx, (labels_per_img, predictions_per_img) in enumerate(zip(labels, predictions)):
            cm = DetectionConfusionMatrix(
                num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5,
                img_idx=img_idx
            )
            cm.process_batch(predictions_per_img, labels_per_img)
            single_confusion_with_img_indices = cm.get_confusion_with_img_indices()
            for i in range(num_classes):
                for j in range(num_classes):
                    confusion_with_img_indices[i][j] += single_confusion_with_img_indices[i][j]
        return confusion_with_img_indices

    def run(
            self,
            num_classes: int,  # include 0th background class
            labels: List[np.ndarray],  # len=images, each: shape=(gt_num, 5), (cid, xmin, ymin, xmax, ymax)
            predictions: List[np.ndarray]  # len=images, each: shape=(pred_num, 6), (xmin, ymin, xmax, ymax, conf, cid)
        ) -> Dict:
        gt_class_cnts = self.get_gt_class_cnts(num_classes, labels)

        out = {}
        out["pr_curves"] = self.get_pr_curves(num_classes, labels, predictions)
        out["refined_pr_curves"] = get_refined_pr_curves(out["pr_curves"])
        out["ap_list"] = get_ap_list(out["refined_pr_curves"])
        out["map"] = get_map(out["ap_list"])
        out["wmap"] = get_wmap(out["ap_list"], gt_class_cnts)

        out["best_threshold"] = ThresholdOptimizer().run(
            out["pr_curves"],
            gt_class_cnts
        )

        out["confusion"] = self.get_confusion(
            num_classes,
            labels,
            predictions,
            out["best_threshold"]
        )
        out["confusion_col_norm"] = get_confusion_axis_norm(out["confusion"], axis=0)
        out["confusion_row_norm"] = get_confusion_axis_norm(out["confusion"], axis=1)
        out["confusion_with_img_indices"] = self.get_confusion_with_img_indices(
            num_classes,
            labels,
            predictions,
            out["best_threshold"]
        )
        out["sorted_accuracy_index_list"] = get_sorted_accuracy_index_list(
            len(labels),
            out["confusion_with_img_indices"]
        )

        out = deserialize(out)
        return out
    