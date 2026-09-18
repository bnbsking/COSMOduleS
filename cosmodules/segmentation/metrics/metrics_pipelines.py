from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn import metrics as skm
from pydantic import BaseModel
from tqdm import tqdm

from cosmodules.segmentation.metrics.confusion_matrix import (
    SegmentationConfusionMatrix
)
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


class SegmentationLabel(BaseModel):
    """
    segmentation_path (str): path to the segmentation mask.
    detection (np.array):
        - instance segmentation only. semantic segmentation use empty numpy array.
        - shape=(gt_num, 5). (cid, xmin, ymin, xmax, ymax)
    """
    model_config = {"arbitrary_types_allowed": True}

    segmentation_path: str
    detection: np.ndarray


class SegmentationPrediction(BaseModel):
    """
    segmentation_path (str): path to the segmentation mask.
    detection (np.array):
        - instance segmentation only. semantic segmentation use empty numpy array.
        - shape=(num_boxes, 6). (xmin, ymin, xmax, ymax, conf, cid)
    """
    model_config = {"arbitrary_types_allowed": True}

    segmentation_path: str
    detection: np.ndarray


class InstanceSegmentationMetricsPipeline:
    @staticmethod
    def get_gt_class_cnts(num_classes: int, labels: List[SegmentationLabel]) -> List[int]:
        gt_class_cnts = [0] * (num_classes - 1)
        for label in labels:
            detection = label.detection
            for i in range(len(detection)):
                gt_class_cnts[detection[i][0] - 1] += 1
        return gt_class_cnts

    @staticmethod
    def get_pr_curves(
            num_classes: int,
            labels: List[SegmentationLabel],
            predictions: List[SegmentationPrediction],
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
            
            for label, prediction in zip(labels, predictions):
                label_mask = np.load(label.segmentation_path, allow_pickle=True)
                prediction_mask = np.load(prediction.segmentation_path, allow_pickle=True)

                img_confusion = SegmentationConfusionMatrix(
                    num_classes,
                    CONF_THRESHOLD = threshold,
                    IOU_THRESHOLD = 0.5
                )
                img_confusion.process_batch(
                    prediction.detection,
                    label.detection,
                    prediction_mask,
                    label_mask
                )
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
            labels: List[SegmentationLabel],
            predictions: List[SegmentationPrediction],
            threshold: float = 0.5,
        ) -> np.ndarray:
        confusion = np.zeros((num_classes, num_classes))  # row: gt, col: pd
        
        for label, prediction in zip(labels, predictions):
            label_mask = np.load(label.segmentation_path, allow_pickle=True)
            prediction_mask = np.load(prediction.segmentation_path, allow_pickle=True)
            cm = SegmentationConfusionMatrix(
                num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5
            )
            cm.process_batch(
                prediction.detection,
                label.detection,
                prediction_mask,
                label_mask
            )
            confusion += cm.get_confusion()

        return confusion

    @staticmethod
    def get_confusion_with_img_indices(
            num_classes: int,
            labels: List[SegmentationLabel],
            predictions: List[SegmentationPrediction],
            threshold: float = 0.5,
        ) -> List[List[Counter[int, int]]]:
        confusion_with_img_indices = [
            [Counter() for _ in range(num_classes)] for _ in range(num_classes)
        ]

        for img_idx, (label, prediction) in enumerate(zip(labels, predictions)):
            label_mask = np.load(label.segmentation_path, allow_pickle=True)
            prediction_mask = np.load(prediction.segmentation_path, allow_pickle=True)
            cm = SegmentationConfusionMatrix(
                num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5,
                img_idx=img_idx
            )
            cm.process_batch(
                prediction.detection,
                label.detection,
                prediction_mask,
                label_mask
            )
            single_confusion_with_img_indices = cm.get_confusion_with_img_indices()
            for i in range(num_classes):
                for j in range(num_classes):
                    confusion_with_img_indices[i][j] += single_confusion_with_img_indices[i][j]
        
        return confusion_with_img_indices

    def run(
            self,
            num_classes: int,  # number of classes, where the first element is "__background__".
            labels: List[SegmentationLabel],
            predictions: List[SegmentationPrediction],
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


class SemanticSegmentationMetricsPipeline:
    @staticmethod
    def get_gt_class_cnts(num_classes: int, labels: List[SegmentationLabel]) -> List[int]:
        gt_class_cnts = [0] * (num_classes - 1)
        for label in labels:
            label = np.load(label.segmentation_path, allow_pickle=True)
            for i in range(1, label.shape[0]):
                gt_class_cnts[i - 1] += int(np.sum(label[i]))
        return gt_class_cnts

    @staticmethod
    def get_pr_curves(
            num_classes: int,
            labels: List[SegmentationLabel],
            predictions: List[SegmentationPrediction],
            k: int = 101
        ) -> List[PRCurve]:
        pr_curves = [
            PRCurve(
                precision=[0.] * k,
                recall=[0.] * k,
            ) for _ in range(num_classes - 1)
        ]

        range_list = list(range(num_classes))
        for i, threshold in tqdm(enumerate(np.linspace(0, 1, k))):
            # update pr curve at the threshold from confusion
            confusion = np.zeros(
                (num_classes, num_classes)
            )  # (i, j) = (gt, pd)
            
            for label, prediction in zip(labels, predictions):
                label_mask = np.load(label.segmentation_path, allow_pickle=True)
                label = label_mask.argmax(axis=0)
                prediction_mask = np.load(prediction.segmentation_path, allow_pickle=True)
                prediction_argmax = prediction_mask.argmax(axis=0)
                prediction = np.where(
                    prediction_mask.max(axis=0) >= threshold, prediction_argmax, 0
                )
                confusion += skm.confusion_matrix(
                    label.reshape(-1),
                    prediction.reshape(-1),
                    labels = range_list
                )
            
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
            labels: List[SegmentationLabel],
            predictions: List[SegmentationPrediction],
            threshold: float = 0.5
        ) -> np.ndarray:
        range_list = list(range(num_classes))
        confusion = np.zeros((num_classes, num_classes))  # row: gt, col: pd
        
        for label_dict, prediction_dict in zip(labels, predictions):
            label_mask = np.load(label_dict.segmentation_path, allow_pickle=True)
            label = label_mask.argmax(axis=0)
            prediction_mask = np.load(prediction_dict.segmentation_path, allow_pickle=True)
            prediction_argmax = prediction_mask.argmax(axis=0)
            
            if num_classes == 2:
                prediction = np.where(
                    prediction_mask.max(axis=0) >= threshold, prediction_argmax, 0
                )
            else:
                prediction = prediction_argmax
            
            confusion += skm.confusion_matrix(
                label.reshape(-1),
                prediction.reshape(-1),
                labels = range_list
            )

        return confusion

    @staticmethod
    def get_confusion_with_img_indices(
            num_classes: int,
            labels: List[SegmentationLabel],
            predictions: List[SegmentationPrediction],
            threshold: float = 0.5,
        ) -> List[List[Counter[int, int]]]:
        range_list = list(range(num_classes))
        confusion_with_img_indices = [
            [Counter() for _ in range(num_classes)] for _ in range(num_classes)
        ]

        for img_idx, (label_dict, prediction_dict) in enumerate(zip(labels, predictions)):
            label_mask = np.load(label_dict.segmentation_path, allow_pickle=True)
            label = label_mask.argmax(axis=0)
            prediction_mask = np.load(prediction_dict.segmentation_path, allow_pickle=True)
            prediction_argmax = prediction_mask.argmax(axis=0)
            
            if num_classes == 2:
                prediction = np.where(
                    prediction_mask.max(axis=0) >= threshold, prediction_argmax, 0
                )
            else:
                prediction = prediction_argmax
            
            confusion = skm.confusion_matrix(
                label.reshape(-1),
                prediction.reshape(-1),
                labels = range_list
            )

            for i in range(num_classes):
                for j in range(num_classes):
                    confusion_with_img_indices[i][j] += Counter({img_idx: int(confusion[i][j])})
        
        return confusion_with_img_indices

    def run(
            self,
            num_classes: int,  # number of classes, where the first element is "__background__".
            labels: List[SegmentationLabel],
            predictions: List[SegmentationPrediction],
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
