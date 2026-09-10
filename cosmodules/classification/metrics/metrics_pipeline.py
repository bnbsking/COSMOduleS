from collections import Counter
from typing import List

import numpy as np
import sklearn.metrics as skm

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


class ClassificationMetricsPipeline:
    @staticmethod
    def get_gt_class_cnts(
            num_classes: int,
            labels: np.ndarray,
            is_single_label: bool,
            start_idx: int = 0
        ) -> List[int]:
        """frontend classes only"""
        gt_class_cnts = [0] * (num_classes - start_idx)
        if is_single_label:
            for label in labels:
                gt_class_cnts[label - start_idx] += 1
        else:
            for label_list in labels:  # multilabel returns positive count only
                for cls_idx, is_positive in enumerate(label_list):
                    gt_class_cnts[cls_idx] += int(is_positive)
        return gt_class_cnts
    
    @staticmethod
    def get_pr_curves(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            k: int = 101,
            is_single_label: bool = True,
            start_idx: int = 0
        ) -> List[PRCurve]:
        pr_curves = [
            PRCurve(
                precision=[0.] * k,
                recall=[0.] * k,
            )
            for _ in range(num_classes - start_idx)
        ]
        for i, threshold in enumerate(np.linspace(0, 1, k)):
            for j in range(start_idx, num_classes):
                if is_single_label:
                    gt_cls = labels
                    pd_cls = (predictions[:, j] >= threshold).astype(np.int32)
                else:
                    gt_cls = labels[:, j]
                    pd_cls = (predictions[:, j, 1] >= threshold).astype(np.int32)
                precision = skm.precision_score(gt_cls, pd_cls, zero_division=0.0)
                recall = skm.recall_score(gt_cls, pd_cls, zero_division=0.0)
                pr_curves[j - start_idx].precision[i] = precision
                pr_curves[j - start_idx].recall[i] = recall
        return pr_curves
    
    @staticmethod
    def get_confusion(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            threshold: float = 0.5,
            is_single_label: bool = True,
            start_idx: int = 0
        ) -> np.ndarray:
        """
        Multi-class classification does not have background, threshold is meaningless.
        """
        if is_single_label and start_idx == 0 and num_classes > 2:  # multiclass without background
            gt_cls = labels
            pd_cls = predictions.argmax(axis=1)
        elif is_single_label:  # binary or "multiclass with background"
            gt_cls = labels
            pd_cls = np.where(
                    predictions[:, 0] < threshold,
                    0,
                    predictions[:, 1:].argmax(axis=1) + 1
                )
        else:  # multilabel
            gt_cls = labels.reshape(-1)
            pd_cls = (predictions[:, :, 1] >= threshold).reshape(-1).astype(np.int32)
        
        confusion = skm.confusion_matrix(gt_cls, pd_cls, labels=list(range(num_classes)))
        return confusion

    @staticmethod
    def get_confusion_with_img_indices(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            threshold: float = 0.5,
            is_single_label: bool = True,
            start_idx: int = 0
        ) -> List[List[Counter[int, int]]]:
        """
        Returns:
            confusion_with_img_indices (List[List[Counter[int, int]]]):
                shape=(num_classes, num_classes). each grid is counter of image indices
        Notes:
            For multi-class classification does not have background, threshold is meaningless.
        """
        if is_single_label and start_idx == 0 and num_classes > 2:  # multiclass without background
            gt_cls = labels
            pd_cls = predictions.argmax(axis=1)
        elif is_single_label:  # binary or "multiclass with background"
            gt_cls = labels
            pd_cls = np.where(
                    predictions[:, 0] < threshold,
                    0,
                    predictions[:, 1:].argmax(axis=1) + 1
                )
        else:  # multilabel
            gt_cls = labels.reshape(-1)
            pd_cls = (predictions[:, :, 1] >= threshold).reshape(-1).astype(np.int32)

        confusion_with_img_indices = [
            [Counter() for _ in range(num_classes)] for _ in range(num_classes)
        ]
        dataset_length = len(labels)
        for idx, (gt_c, pd_c) in enumerate(zip(gt_cls, pd_cls)):
            confusion_with_img_indices[gt_c][pd_c][idx % dataset_length] += 1
        return confusion_with_img_indices


    def run(
            self,
            num_classes: int,  # include 0th background class if exist
            labels: np.ndarray,  # single label: shape=(data,); multi-label: shape=(data, multi-label-dim)
            predictions: np.ndarray,  # single label: shape=(data, num_classes); multi-label: shape=(data, multi-label-dim, 2)
            start_idx: int = 0,  # normal: 0; background: 1
        ):
        """
        + elements in multi-label must be 0 or 1
        + multi-label does not have a background class
        + background class must starts from 0, not limited to 1 background class 
        """
        is_single_label = bool(len(labels.shape) == 1)
        gt_class_cnts = self.get_gt_class_cnts(num_classes, labels, is_single_label, start_idx)
        
        out = {}
        out["pr_curves"] = self.get_pr_curves(num_classes, labels, predictions, 101, is_single_label, start_idx)
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
            out["best_threshold"],
            is_single_label,
            start_idx
        )
        out["confusion_col_norm"] = get_confusion_axis_norm(out["confusion"], axis=0)
        out["confusion_row_norm"] = get_confusion_axis_norm(out["confusion"], axis=1)
        out["confusion_with_img_indices"] = self.get_confusion_with_img_indices(
            num_classes,
            labels,
            predictions,
            out["best_threshold"],
            is_single_label,
            start_idx
        )
        out["sorted_accuracy_index_list"] = get_sorted_accuracy_index_list(
            len(labels),
            out["confusion_with_img_indices"]
        )

        out = deserialize(out)
        return out
        