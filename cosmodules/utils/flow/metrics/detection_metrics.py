from typing import Dict, List
from collections import Counter

import numpy as np
import sklearn.metrics as skm
from tqdm import tqdm

from cosmodules.utils.flow.metrics.base_metrics import BaseMetricsFlow, PRCurve, Confusion, ThresholdOptimizer
from cosmodules.utils.detection.confusion_matrix import DetectionConfusionMatrix


class DetectionMetricsFlow(BaseMetricsFlow):
    def __init__(
            self,
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            thres_optim: ThresholdOptimizer = ThresholdOptimizer("f1"),
            save_path: str = ""
        ):
        """
        Given basic arguments, self.run iterates func_dicts and save all metrics as results.
        Args:
            num_classes (int): number of classes, where the first element is "__background__".
            labels (List[np.ndarray]): length is the number of images. each numpy has shape (N, 5).
                N is the number of ground truth, and 5 refers to (cid, xmin, ymin, xmax, ymax)
            predictions (List[np.ndarray]): length is the number of images. each numpy has shape (M, 6).
                M is the number of predictions, and 6 refers to (xmin, ymin, xmax, ymax, conf, cid)
            func_dicts (List[Dict]): length is the number of metrics function.
                each dict has the format {"func_name": str, "args": Dict, "log_name": str}.
                self.run saves the output in self.metrics, where log_name is key and output is value
            save_path (str): path to save the result as json
        """
        super().__init__(num_classes, labels, predictions, thres_optim, save_path)
    
    @staticmethod
    def get_gt_class_cnts(num_classes: int, labels: List[np.ndarray]) -> List[int]:
        gt_class_cnts = [0] * (num_classes - 1)
        for label in labels:
            for i in range(len(label)):
                gt_class_cnts[label[i][0] - 1] += 1
        return gt_class_cnts

    @staticmethod
    def get_pr_curves(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            k: int = 101
        ) -> List[Dict[str, List[float]]]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(num_classes - 1)
        ]

        for i, threshold in tqdm(enumerate(np.linspace(0, 1, k))):
            # get confusion of the threshold
            confusion = np.zeros(
                (num_classes, num_classes)
            )  # (i, j) = (gt, pd)

            for labels_, predictions_ in zip(labels, predictions):
                img_confusion = DetectionConfusionMatrix(
                    num_classes,
                    CONF_THRESHOLD = threshold,
                    IOU_THRESHOLD = 0.5
                )
                img_confusion.process_batch(predictions_, labels_)
                confusion += img_confusion.get_confusion()
            
            # update pr curve at the threshold from confusion
            row_sum = confusion.sum(axis=1)
            col_sum = confusion.sum(axis=0)
            for cid in range(1, num_classes):
                pr_curves[cid-1]["precision"][i] = confusion[cid][cid] / col_sum[cid] if col_sum[cid] else 0
                pr_curves[cid-1]["recall"][i] = confusion[cid][cid] / row_sum[cid] if row_sum[cid] else 0

        return [PRCurve(**pr_curve) for pr_curve in pr_curves]
    
    @staticmethod
    def get_confusion(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            threshold: float = 0.5,
        ) -> np.ndarray:
        confusion = np.zeros((num_classes, num_classes))  # row: gt, col: pd
        for labels_, predictions_ in zip(labels, predictions):
            cm = DetectionConfusionMatrix(
                num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5
            )
            cm.process_batch(predictions_, labels_)
            confusion += cm.get_confusion()
        return Confusion(confusion)
    
    @staticmethod
    def get_confusion_with_img_indices(
            num_classes: int,
            labels: np.ndarray,
            predictions: np.ndarray,
            threshold: float = 0.5,
        ) -> List[List[Counter[int, int]]]:
        """
        Returns:
            confusion_with_img_indices (List[List[Counter[int, int]]]):
                shape=(num_classes, num_classes). each grid is Counters (img_idx -> cnts) 
        """
        confusion_with_img_indices = [
            [Counter() for _ in range(num_classes)] for _ in range(num_classes)
        ]
        for img_idx, (labels_, predictions_) in enumerate(zip(labels, predictions)):
            cm = DetectionConfusionMatrix(
                num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5,
                img_idx=img_idx
            )
            cm.process_batch(predictions_, labels_)
            single_confusion_with_img_indices = cm.get_confusion_with_img_indices()
            for i in range(num_classes):
                for j in range(num_classes):
                    confusion_with_img_indices[i][j] += single_confusion_with_img_indices[i][j]
        return confusion_with_img_indices
