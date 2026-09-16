import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]  # background first


def prediction_probs_to_cids(
        pd_filled_probs_npy: np.ndarray,
        pd_boxes: None | np.ndarray = None,
        pd_probs: None | np.ndarray = None,
    ) -> np.ndarray:
    """
    Args:
        pd_filled_probs_npy (np.ndarray): shape=(categories, h, w) and values are floats in [0, 1]
        pd_boxes (None | np.ndarray): shape=(objects, 4) and each row is [xmin, ymin, xmax, ymax]
        pd_probs (None | np.ndarray): shape=(n, categories) and each row is probabilities
    Returns:
        pd_filled_cids_npy (np.ndarray): shape=(h, w) and values are in {0, 1, 2, ..., categories - 1}
    Notes:
        For instance segmentation, "pd_boxes" and "pd_probs" are required.
        For semantic segmentation, "pd_boxes" and "pd_probs" must be None.
    """
    if pd_boxes is not None:
        pd_filled_cids_npy = np.zeros(pd_filled_probs_npy.shape[1:], dtype=np.uint8)
        for pd_box, pd_prob in zip(pd_boxes, pd_probs):
            xmin, ymin, xmax, ymax = pd_box
            box_cls = np.argmax(pd_prob)
            # force pixels align to the box
            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    pixel_cls = pd_filled_probs_npy[:, y, x].argmax()
                    if pixel_cls == box_cls:
                        pd_filled_cids_npy[y, x] = box_cls
                    else:
                        pd_filled_cids_npy[y, x] = 0
    else:
        pd_filled_cids_npy = np.argmax(pd_filled_probs_npy, axis=0)
    return pd_filled_cids_npy


def mask2contour(mask: np.ndarray, contour_width: int = 2) -> np.ndarray:
    """
    Args:
        mask (np.ndarray): shape=(h, w) and values are in {0, 1, 2, ..., categories - 1}
    Returns:
        contour (np.ndarray): shape=(h, w) and values are in {0, 1, 2, ..., categories - 1}
    """
    contour_mask = np.zeros_like(mask).astype(np.uint8)
    for category in np.unique(mask):
        if category == 0:
            continue
        binary_mask = (mask == category).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_mask, contours, -1, int(category), contour_width)
    return contour_mask


def merge_img_with_contour(
        img: np.ndarray,
        contour: np.ndarray
    ) -> np.ndarray:
    """
    Args:
        img (np.ndarray): shape=(h, w, 3) and values are in [0, 255]
        contour (np.ndarray): shape=(h, w) and values are in {0, 1, 2, ..., categories - 1}
    Returns:
        img_seg (np.ndarray): shape=(h, w, 3) and values are in [0, 255]
    """
    img_seg = img.copy()
    for category_id in np.unique(contour):
        if category_id > 0:
            color = colors[category_id % len(colors)]
            indices_y, indices_x = np.where(contour == category_id)
            for x, y in zip(indices_x, indices_y):
                for channel in range(3):
                    img_seg[y, x, channel] = color[channel] * 255
    return img_seg


def show_semantic_mask(
        categories: List[str],
        img: np.ndarray,
        gt_contour_npy: np.ndarray,
        pd_contour_npy: None | np.ndarray = None,
        save_path: None | str = None,
    ):
    """
    Core of the visualization.
    Args:
        categories (List[str]): length is max_category_id
        img (np.ndarray): shape=(h, w, 3) and values are in [0, 255]
        gt_contour_npy (np.ndarray): shape=(h, w) and values are in {0, 1, 2, ..., categories - 1}
        pd_contour_npy (None | np.ndarray): shape=(h, w) and values are in {0, 1, 2, ..., categories - 1}
        save_path: str
    """
    categories.pop(0) if categories[0] == "__background__" else None

    img_seg_gt = merge_img_with_contour(img[:, :, ::-1], gt_contour_npy)
    if pd_contour_npy is not None:
        img_seg_pd = merge_img_with_contour(img[:, :, ::-1], pd_contour_npy)
    else:
        img_seg_pd = img
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("GT")
    plt.imshow(img_seg_gt)

    plt.subplot(1, 2, 2)
    plt.title("PD")
    plt.imshow(img_seg_pd)
    for i in range(len(categories)):
        plt.scatter([], [], color=colors[i+1], label=categories[i])
    plt.legend(labels=categories)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()
