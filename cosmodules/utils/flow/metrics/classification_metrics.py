from typing import List
from collections import Counter

import numpy as np
import sklearn.metrics as skm

from cosmodules.utils.flow.metrics.base_metrics import BaseMetricsFlow, PRCurve, Confusion, ThresholdOptimizer


class ClassificationMetricsFlow(BaseMetricsFlow):
    def __init__(
            self,
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            thres_optim: ThresholdOptimizer = ThresholdOptimizer("f1"),
            save_path: str = "",
            start_idx: int = 0,
        ):
        """
        Given basic arguments, self.run iterates func_dicts and save all metrics as results.
        Args:
            num_classes (int): number of classes (includes background if exist).
            labels (np.ndarray):
                For single label, shape=(data,)
                For multi label, shape=(data, multi-label-dim).
            predictions (np.ndarray):
                For single label, shape=(data, num_classes). If background exist, it must be zeroth category. 
                For multi label, shape=(data, multi-label-dim, 2)
            func_dicts (List[Dict]): length is the number of metrics function.
                each dict has the format {"func_name": str, "args": Dict, "log_name": str}.
                self.run saves the output in self.metrics, where log_name is key and output is value
            save_path (str): path to save the result as json
            start_idx (int): be 0 or 1 (has background)
        """
        super().__init__(num_classes, labels, predictions, thres_optim, save_path)
        self.single_label = len(labels.shape) == 1
        self.start_idx = start_idx if self.single_label else 0

    @staticmethod
    def get_gt_class_cnts(num_classes: int, labels: np.ndarray, single_label: bool, start_idx: int = 0) -> List[int]:
        gt_class_cnts = [0] * (num_classes - start_idx)
        if single_label:
            for label in labels:
                gt_class_cnts[label - start_idx] += 1
        else:
            for label_list in labels:  # multilabel returns positive count only
                for cls_idx, is_positive in enumerate(label_list):
                    gt_class_cnts[cls_idx] += int(is_positive)
        return gt_class_cnts
    
    def _set_gt_class_cnts(self):
        self.metrics["gt_class_cnts"] = self.get_gt_class_cnts(
            self.num_classes, self.labels, self.single_label, self.start_idx
        )

    @staticmethod
    def get_pr_curves(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            k: int = 101,  # the granularity of thresholds. i.e. Scanning thresholds within np.linspace(0,1,k) 
            single_label: bool = True,
            start_idx: int = 0
        ) -> List[PRCurve]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(num_classes - start_idx)
        ]
        for i, threshold in enumerate(np.linspace(0, 1, k)):
            for j in range(start_idx, num_classes):
                if single_label:
                    gt_cls = labels
                    pd_cls = (predictions[:, j] >= threshold).astype(np.int32)
                else:
                    gt_cls = labels[:, j]
                    pd_cls = (predictions[:, j, 1] >= threshold).astype(np.int32)
                precision = skm.precision_score(gt_cls, pd_cls, zero_division=0.0)
                recall = skm.recall_score(gt_cls, pd_cls, zero_division=0.0)
                pr_curves[j - start_idx]["precision"][i] = precision
                pr_curves[j - start_idx]["recall"][i] = recall
        return [PRCurve(**pr_curve) for pr_curve in pr_curves]
    
    def _set_pr_curves(self, k = 101):
        pr_curves = self.get_pr_curves(
            self.num_classes, self.labels, self.predictions, k, self.single_label, self.start_idx
        )
        self.metrics["pr_curves"] = [pr_curve.run() for pr_curve in pr_curves]
    
    @staticmethod
    def get_confusion(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            threshold: float = 0.5,
            single_label: bool = True,
            start_idx: int = 0
        ) -> Confusion:
        """
        Notes:
            For multi-class classification does not have background, threshold is meaningless.
        """
        if single_label and start_idx==0 and num_classes > 2:  # multiclass without background
            gt_cls = labels
            pd_cls = predictions.argmax(axis=1)
        elif single_label:  # binary or "multiclass with background"
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
        return Confusion(confusion)
    
    def _set_confusion(self) -> None:
        confusion = self.get_confusion(
            self.num_classes, self.labels, self.predictions, self.metrics["threshold"], self.single_label, self.start_idx
        )
        self.metrics["confusion"] = confusion.run()

    @staticmethod
    def get_confusion_with_img_indices(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            threshold: float = 0.5,
            single_label: bool = True,
            start_idx: int = 0
        ) -> List[List[Counter[int, int]]]:
        """
        Returns:
            confusion_with_img_indices (List[List[Counter[int, int]]]):
                shape=(num_classes, num_classes). each grid is counter of image indices
        Notes:
            For multi-class classification does not have background, threshold is meaningless.
        """
        if single_label and start_idx==0 and num_classes > 2:  # multiclass without background
            gt_cls = labels
            pd_cls = predictions.argmax(axis=1)
        elif single_label:  # binary or "multiclass with background"
            gt_cls = labels
            pd_cls = np.where(
                    predictions[:, 0] > threshold,
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
        for idx, (gtc, pdc) in enumerate(zip(gt_cls, pd_cls)):
            confusion_with_img_indices[gtc][pdc][idx % dataset_length] += 1
        return confusion_with_img_indices

    def _set_confusion_with_img_indices(self):
        self.metrics["confusion_with_img_indices"] = self.get_confusion_with_img_indices(
            self.num_classes, self.labels, self.predictions, self.metrics["threshold"], self.single_label, self.start_idx
        )