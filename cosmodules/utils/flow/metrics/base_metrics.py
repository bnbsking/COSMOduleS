from abc import abstractmethod
from collections import Counter
import json
import os
from typing import Callable, Dict, List, Tuple, Union

import numpy as np


class PRCurve:
    """PR curve of a class where len(precision) == len(recall) == k (101 by default)"""
    def __init__(self, precision: List[float], recall: List[float]):
        assert len(precision) == len(recall)
        self.precision = precision
        self.recall = recall
        #
        self.refine_precision: List[float] = []
        self.refine_recall: List[float] = []
        self.ap: float = 0.0

    @staticmethod
    def get_refine(precision: List[float], recall: List[float]) -> Tuple[List[float], List[float]]:
        zip_arr = sorted(zip(recall, precision))
        recall_arr, precision_arr = zip(*zip_arr)
        recall_arr, precision_arr = list(recall_arr), list(precision_arr)
        for j in range(1, len(precision_arr)):
            precision_arr[-1-j] = max(precision_arr[-1-j], precision_arr[-j])
        return recall_arr, precision_arr

    def _set_refine(self) -> None:
        self.refine_recall, self.refine_precision = self.get_refine(self.precision, self.recall)
    
    @staticmethod
    def get_ap(refine_precision: List[float], refine_recall: List[float]) -> float:
        ap = 0
        for j in range(len(refine_precision) - 1):  # 101 - 1
            ap += refine_precision[j] * (refine_recall[j+1] - refine_recall[j])
        return ap

    def _set_ap(self) -> None:
        self.ap = self.get_ap(self.refine_precision, self.refine_recall)
    
    def run(self) -> Dict:
        self._set_refine()
        self._set_ap()
        return {
            "precision": self.precision,
            "recall": self.recall,
            "refine_precision": self.refine_precision,
            "refine_recall": self.refine_recall,
            "ap": round(self.ap, 3),
        }


class Confusion:
    """
    Confusion matrix with shape=(num_classes, num_classes)
    C[row][col] means GT=row, pred=col
    1. Precision of class i = C[i][i] / sum(C[:, i])  # col norm; axis=0
    2. Recall of class i = C[i][i] / sum(C[i, :])  # row norm; axis=1
    """
    def __init__(self, confusion: np.ndarray):  # shape=(num_classes, num_classes)
        assert confusion.shape[0] == confusion.shape[1]
        self.confusion = confusion
        #
        self.confusion_col_norm: np.ndarray = None
        self.confusion_row_norm: np.ndarray = None

    @staticmethod
    def get_confusion_axis_norm(confusion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        col_sum = confusion.sum(axis=0)
        row_sum = confusion.sum(axis=1)
        confusion_col_norm = confusion.astype(float)
        confusion_row_norm = confusion.astype(float)
        for i in range(len(col_sum)):
            confusion_col_norm[:, i] /= (col_sum[i] + 1e-10)
            confusion_row_norm[i, :] /= (row_sum[i] + 1e-10)
        return confusion_col_norm, confusion_row_norm

    def _set_confusion_axis_norm(self) -> None:
        self.confusion_col_norm, self.confusion_row_norm = self.get_confusion_axis_norm(self.confusion)
    
    def run(self) -> Dict:
        self._set_confusion_axis_norm()
        return {
            "confusion": self.confusion.tolist(),
            "confusion_col_norm": self.confusion_col_norm.tolist(),
            "confusion_row_norm": self.confusion_row_norm.tolist(),
        }


class ThresholdOptimizer:
    def __init__(self, strategy: Union[str, Callable] = "f1"):
        self.func = getattr(self, f"_strategy_{strategy}") if isinstance(strategy, str) else strategy

    def _strategy_half(self, precision: float, recall: float) -> float:
        score = np.zeros(len(precision))
        score[len(score) // 2] = 1.0
        return score

    def _strategy_f1(self, precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall + 1e-10)
    
    def _strategy_precision(self, precision: float, recall: float) -> float:
        return np.where(recall >= 0.5, precision, 0)
    
    def run(self, pr_curves: List[Dict], gt_class_cnts: List[int] = None) -> float:
        """
        Args:
            pr_curves: same format as output of pr_curves.run()
        Notes:
            For classification does not have background, the return value is meaningless.
        """
        if gt_class_cnts is None:
            gt_class_cnts = [1] * len(pr_curves)
        k_val = len(pr_curves[0]["precision"])
        thresholds = np.linspace(0, 1, k_val)  # 101
        weighted_score = [0] * len(thresholds)
        for class_i, pr_curve in enumerate(pr_curves):
            for k_idx, (precision, recall) in enumerate(zip(pr_curve["precision"], pr_curve["recall"])):
                score = self.func(precision, recall)
                weighted_score[k_idx] += score * gt_class_cnts[class_i] / sum(gt_class_cnts)
        _, best_threshold = max(zip(weighted_score, thresholds))
        return float(best_threshold)


class BaseMetricsFlow:
    def __init__(
            self,
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            thres_optim: ThresholdOptimizer = ThresholdOptimizer("f1"),
            save_path: str = "",
        ):
        self.num_classes = num_classes
        self.labels = labels
        self.predictions = predictions
        self.thres_optim = thres_optim
        self.save_path = save_path
        #
        self.metrics = {
            "gt_class_cnts": [],  # List[int], length = self.num_classes or self.num_classes - 1
            "pr_curves": [],  # List[PRCurve], length = self.num_classes or self.num_classes - 1
            "map": -1,
            "wmap": -1,
            "confusion": None,  # np.ndarray with shape=(num_classes, num_classes)
        }

    @staticmethod
    @abstractmethod
    def get_gt_class_cnts(num_classes: int, labels: np.ndarray) -> List[int]:
        pass

    def _set_gt_class_cnts(self) -> None:
        self.metrics["gt_class_cnts"] = self.get_gt_class_cnts(self.num_classes, self.labels)

    @staticmethod
    @abstractmethod
    def get_pr_curves(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            k: int = 101  # the granularity of thresholds. i.e. Scanning thresholds within np.linspace(0,1,k) 
        ) -> List[PRCurve]:
        pass

    def _set_pr_curves(self, k: int = 101) -> None:
        pr_curves = self.get_pr_curves(self.num_classes, self.labels, self.predictions, k)
        self.metrics["pr_curves"] = [pr_curve.run() for pr_curve in pr_curves]

    def _set_ap_list(self) -> None:
        ap_list = [pr_curve_dict["ap"] for pr_curve_dict in self.metrics["pr_curves"]]
        self.metrics["ap_list"] = ap_list

    def _set_map(self) -> None:
        ap_list = self.metrics["ap_list"]
        self.metrics["map"] = round(sum(ap_list) / len(ap_list), 3)
    
    def _set_wmap(self) -> None:
        ap_list = self.metrics["ap_list"]
        gt_class_cnts = self.metrics["gt_class_cnts"]
        self.metrics["wmap"] = round(sum(ap * cnt for ap, cnt in zip(ap_list, gt_class_cnts)) / sum(gt_class_cnts), 3)

    def _set_threshold(self) -> None:
        self.metrics["threshold"] = self.thres_optim.run(
            self.metrics["pr_curves"],
            self.metrics["gt_class_cnts"]
        )

    @staticmethod
    @abstractmethod
    def get_confusion(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            threshold: float = 0.5
        ) -> Confusion:
        pass

    def _set_confusion(self) -> None:
        confusion = self.get_confusion(
            self.num_classes, self.labels, self.predictions, self.metrics["threshold"]
        )
        self.metrics["confusion"] = confusion.run()

    @staticmethod
    @abstractmethod
    def get_confusion_with_img_indices(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            threshold: float = 0.5,
        ) -> List[List[Counter[int, int]]]:
        """For single label classification, the counter key (img id) is unique in a certain cell,
        and its value always be 1"""
        pass

    def _set_confusion_with_img_indices(self) -> None:
        self.metrics["confusion_with_img_indices"] = self.get_confusion_with_img_indices(
            self.num_classes, self.labels, self.predictions, self.metrics["threshold"]
        )

    def _deserialize(self, data: Dict):
        if isinstance(data, dict):
            return {k: self._deserialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            metrics = self._deserialize(self.metrics)
            json.dump(metrics, f, indent=4)

    def run(self) -> Dict:
        self._set_gt_class_cnts()
        self._set_pr_curves()
        self._set_ap_list()
        self._set_map()
        self._set_wmap()
        self._set_threshold()
        self._set_confusion()
        self._set_confusion_with_img_indices()
        if self.save_path:
            self.save()
        return self.metrics
    