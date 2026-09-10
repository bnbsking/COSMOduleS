import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel


colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]  # background first
seven_segment_display_dict = {
    0: [1, 0, 1, 1, 1, 1, 1],
    1: [0, 0, 0, 1, 0, 1, 0],
    2: [1, 1, 1, 0, 1, 1, 0],
    3: [1, 1, 1, 0, 1, 0, 1],
    4: [0, 1, 0, 1, 1, 0, 1],
    5: [1, 1, 1, 1, 0, 0, 1],
    6: [1, 1, 1, 1, 0, 1, 1],
    7: [1, 0, 0, 0, 1, 0, 1],
    8: [1, 1, 1, 1, 1, 1, 1],
    9: [1, 1, 1, 1, 1, 0, 1],
    10: [],  # percent
}  # digit -> [up, mid, down, upleft, upright, downleft, downright]


def get_digit_patch(digit: int, color = (1, 1, 1)) -> np.ndarray:
    patch = np.array([[color for _ in range(10)] for _ in range(20)]).astype(float)  # zero padding

    seg_list = seven_segment_display_dict[digit]
    if not seg_list:
        patch[:3, :3, :] = 0, 0, 0
        patch[-3:, -3:, :] = 0, 0, 0
        for i in range(10):
            patch[i*2:i*2+2, 10-i-1:10-i, :] = 0, 0, 0
    else:
        if seg_list[0]:
            patch[0:3, :, :] = 0, 0, 0
        if seg_list[1]:
            patch[10-1:10+1, :, :] = 0, 0, 0        
        if seg_list[2]:
            patch[20-3:20, :, :] = 0, 0, 0
        if seg_list[3]:
            patch[:10, :3, :] = 0, 0, 0
        if seg_list[4]:
            patch[:10, 10-3:, :] = 0, 0, 0
        if seg_list[5]:
            patch[10:, :3, :] = 0, 0, 0
        if seg_list[6]:
            patch[10:, 10-3:,:] = 0, 0, 0
    digit_patch = np.array([[color for _ in range(20)] for _ in range(30)]).astype(float)
    digit_patch[5:25, 5:15, :] = patch
    return digit_patch


def get_confidence_patch(unit_digit: int, tens_digit: int, color = (1, 1, 1)) -> np.ndarray:
    digit_patch = np.array([[color for _ in range(60)] for _ in range(30)]).astype(float)
    digit_patch[:, :20, :] = get_digit_patch(unit_digit, color)
    digit_patch[:, 20:40, :] = get_digit_patch(tens_digit, color)
    digit_patch[:, 40:60, :] = get_digit_patch(10, color)
    return digit_patch


class DetectionLabel(BaseModel):
    img_path: str
    gt_boxes: List[Tuple[int, int, int, int]]  # voc
    gt_cls: List[int]
    pd_boxes: List[Tuple[int, int, int, int]]
    pd_probs: List[List[float]]


def show(
        class_list: List[str],
        label: DetectionLabel,
        save_path: str | None = None,
        box_width: int = 4,
        value_ratios: Tuple[int, int] = (1,1)
    ):
    class_list.pop(0) if class_list[0] == "__background__" else None
    img_raw = cv2.imread(label.img_path)[:, :, ::-1]/255

    # ground truth
    img_gt = img_raw.copy()
    boxes_gt = label.gt_boxes 
    cids_gt = label.gt_cls
    for (xmin, ymin, xmax, ymax), cid in zip(boxes_gt, cids_gt):
        img_gt[ymin-box_width:ymin+box_width, xmin:xmax, :] = colors[cid]
        img_gt[ymax-box_width:ymax+box_width, xmin:xmax, :] = colors[cid]
        img_gt[ymin:ymax, xmin-box_width:xmin+box_width, :] = colors[cid]
        img_gt[ymin:ymax, xmax-box_width:xmax+box_width, :] = colors[cid]
    
    # prediction
    img_pd = img_raw.copy()
    pd_probs = label.pd_probs
    pd_boxes = label.pd_boxes

    if pd_probs:
        pd_confs = np.array(pd_probs).max(axis=1)
        pd_cids = np.array(pd_probs).argmax(axis=1)
    else:
        pd_confs = []
        pd_cids = []

    for pd_conf, (xmin, ymin, xmax, ymax), pd_cid in sorted(zip(pd_confs, pd_boxes, pd_cids)):  # plot least conf first
        img_pd[ymin-box_width:ymin+box_width, xmin:xmax, :] = colors[pd_cid]
        img_pd[ymax-box_width:ymax+box_width, xmin:xmax, :] = colors[pd_cid]
        img_pd[ymin:ymax, xmin-box_width:xmin+box_width, :] = colors[pd_cid]
        img_pd[ymin:ymax, xmax-box_width:xmax+box_width, :] = colors[pd_cid]
        
        # confidence patches
        unit_digit, tens_digit = int(pd_conf * 10), int(pd_conf * 100) % 10
        P = get_confidence_patch(unit_digit, tens_digit, color=colors[pd_cid])
        (ph, pw, _), (rh, rw) = P.shape, value_ratios
        P = cv2.resize(P, (int(pw * rw), int(ph * rh)) )
        try:
            if ymin >= P.shape[0] and xmin + P.shape[1] < img_pd.shape[1]:  # upper bar - up
                img_pd[ymin - P.shape[0]:ymin, xmin:xmin + P.shape[1], :] = P
            elif ymax + P.shape[0] < img_pd.shape[0] and xmin + P.shape[1] < img_pd.shape[1]:  # down bar - down
                img_pd[ymax:ymax + P.shape[0], xmin:xmin + P.shape[1], :] = P
            elif ymin + P.shape[0] < img_pd.shape[0] and xmin + P.shape[1]<img_pd.shape[1]:
                img_pd[ymin:ymin+P.shape[0], xmin:xmin + P.shape[1], :] = P  # upper bar - down
            elif ymax + P.shape[0] > 0 and xmin + P.shape[1]<img_pd.shape[1]:  # down bar - up
                img_pd[ymax - P.shape[0]:ymax, xmin:xmin + P.shape[1], :] = P
        except:
            pass

    # plot
    fig = plt.figure(figsize=(20, 10))
    fig.set_facecolor("white")

    plt.subplot(1, 2, 1)
    plt.title("GT", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    for r, g, b in colors[1:]:
        c2hex = lambda c: hex(int(c * 255))[2:].zfill(2)
        plt.scatter([], [], c=f"#{c2hex(r)}{c2hex(g)}{c2hex(b)}")

    plt.legend(labels=class_list, fontsize=16)
    plt.imshow(img_gt)
    
    plt.subplot(1, 2, 2)
    plt.title("Pred", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.imshow(img_pd)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()
