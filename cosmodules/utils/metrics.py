from collections import Counter

from typing import List, Tuple

import numpy as np
from pydantic import BaseModel


class PRCurve(BaseModel):
    precision: List[float]
    recall: List[float]
    has_refined: bool = False


def get_refined_pr_curves(pr_curves: List[PRCurve]) -> List[PRCurve]:
    """
    sorted by recall, and enhance precision by next element reversely
    """
    refined_pr_curves = []
    for pr_curve in pr_curves:
        recall_arr = pr_curve.recall.copy()
        precision_arr = pr_curve.precision.copy()
        zip_arr = sorted(zip(recall_arr, precision_arr))
        recall_arr, precision_arr = zip(*zip_arr)
        recall_arr, precision_arr = list(recall_arr), list(precision_arr)
        for j in range(1, len(precision_arr)):  # enhance precision by next element reversely
            precision_arr[-1-j] = max(precision_arr[-1-j], precision_arr[-j])
        refined_pr_curve = PRCurve(precision=precision_arr, recall=recall_arr, has_refined=True)
        refined_pr_curves.append(refined_pr_curve)
    return refined_pr_curves


def get_ap_list(refined_pr_curves: List[PRCurve]) -> List[float]:
    k_val = len(refined_pr_curves[0].precision)  # 101
    ap_list = []
    for refined_pr_curve in refined_pr_curves:
        ap = 0
        for j in range(k_val - 1):
            ap += refined_pr_curve.precision[j] * \
                (refined_pr_curve.recall[j + 1] - refined_pr_curve.recall[j])
        ap_list.append(round(ap, 3))
    return ap_list


def get_map(ap_list: List[float]) -> float:
    return round(sum(ap_list) / len(ap_list), 3)


def get_wmap(ap_list: List[float], gt_class_cnts: List[int]) -> float:
    return round(sum(ap * cnt for ap, cnt in zip(ap_list, gt_class_cnts)) \
            / sum(gt_class_cnts), 3)


class ThresholdOptimizer:
    @staticmethod
    def f1(precision: float, recall: float, **kwargs) -> float:
        return 2 * precision * recall / (precision + recall + 1e-10)

    @staticmethod
    def precision(precision: float, recall: float, recall_lb: float = 0.5, **kwargs) -> float:
        return precision if recall >= recall_lb else 0

    @staticmethod
    def recall(precision: float, recall: float, precision_lb: float = 0.5, **kwargs) -> float:
        return recall if precision >= precision_lb else 0

    @staticmethod
    def dummy(threshold: float = 0.5, **kwargs) -> float:
        return threshold

    def run(
            self,
            pr_curves: List[PRCurve],
            gt_class_cnts: List[int] | None = None,
            strategy: str = "f1",
            **kwargs
        ) -> float:
        score_func = getattr(self, strategy)

        k_val = len(pr_curves[0].precision)  # 101
        weighted_score_list = [0] * k_val
        if gt_class_cnts is None:
            class_weights = [1 / len(pr_curves)] * len(pr_curves)
        else:
            class_weights = [cnt / sum(gt_class_cnts) for cnt in gt_class_cnts]
        for pr_curve, class_weight in zip(pr_curves, class_weights):
            for j, (precision, recall) in enumerate(
                    zip(pr_curve.precision, pr_curve.recall)
                ):
                args = {"precision": precision, "recall": recall} | kwargs
                score = score_func(**args)
                weighted_score_list[j] += score * class_weight
        thresholds = np.linspace(0, 1, k_val)
        _, best_threshold = max(zip(weighted_score_list, thresholds))
        return best_threshold


def get_confusion_axis_norm(confusion: np.ndarray[np.int64], axis: int) -> np.ndarray[np.float64]:
    """
    e.g.
        input
            confusion: shape = (num_classes, num_classes)
            confusion[i, j]  # gt=i, pred=j
        output
            if axis == 0, normalize each column -> precision
            if axis == 1, normalize each row -> recall
    """
    confusion_axis_norm = confusion.copy()
    axis_sum = confusion_axis_norm.sum(axis=axis)
    confusion_axis_norm = confusion_axis_norm.astype(float)
    for i in range(len(confusion_axis_norm)):
        if axis == 0:
            confusion_axis_norm[:, i] /= (axis_sum[i] + 1e-10)
        elif axis == 1:
            confusion_axis_norm[i, :] /= (axis_sum[i] + 1e-10)
    return confusion_axis_norm


def get_sorted_accuracy_index_list(
        samples: int,
        confusion_with_img_indices: List[List[Counter[int, int]]]
    ) -> List[Tuple[float, int]]:
    right_cnt = [0] * samples
    wrong_cnt = [0] * samples
    for i in range(len(confusion_with_img_indices)):
        for j in range(len(confusion_with_img_indices)):
            for idx, times in confusion_with_img_indices[i][j].items():
                if i == j:
                    right_cnt[idx] += times
                else:
                    wrong_cnt[idx] += times
    accuracy_index_list = [
        (round(rc / (rc + wc + 1e-10), 3), i)
        for i, (rc, wc) in enumerate(zip(right_cnt, wrong_cnt))
    ]
    return sorted(accuracy_index_list)
