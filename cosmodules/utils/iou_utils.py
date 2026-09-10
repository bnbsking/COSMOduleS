from typing import Literal

import numpy as np

from cosmodules.detection.format_conversion.dataset_conversion import (
    ConvertBoxFromAnyToVOC
)


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def get_iou(xmin1: int, ymin1: int, xmax1: int, ymax1: int, xmin2: int, ymin2: int, xmax2: int, ymax2: int) -> float:
    inter = max(0, min(ymax1, ymax2) - max(ymin1, ymin2)) * max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    areaA = (ymax1 - ymin1) * (xmax1 - xmin1)
    areaB = (ymax2 - ymin2) * (xmax2 - xmin2)
    return inter / (areaA + areaB - inter)


def nms_filter(
        bboxes: np.ndarray,  # shape=(N, 4)
        src_type: Literal["voc", "yolo", "coco"],
        threshold=0.3
    ) -> np.ndarray:  # shape=(M, 4)
    alive = set(range(len(bboxes)))
    results = []
    while len(alive) >= 2:
        min_alive = min(alive)
        results.append(min_alive)
        xmin1, ymin1, xmax1, ymax1 = ConvertBoxFromAnyToVOC().run(
            src_type,
            bboxes[min_alive][0],
            bboxes[min_alive][1],
            bboxes[min_alive][2],
            bboxes[min_alive][3],
            img_width = 1000,
            img_height = 1000
        )
        alive.remove(min_alive)
        for idx in alive.copy():
            xmin2, ymin2, xmax2, ymax2 = ConvertBoxFromAnyToVOC().run(
                src_type,
                bboxes[idx][0],
                bboxes[idx][1],
                bboxes[idx][2],
                bboxes[idx][3],
                img_width = 1000,
                img_height = 1000
            )
            iou  = get_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
            if iou >= threshold:
                alive.remove(idx)
    if len(alive)==1:
        results.append(alive.pop())
    return np.array(results)
