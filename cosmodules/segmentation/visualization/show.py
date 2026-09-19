import importlib
import json
import os

import cv2
import numpy as np
from cosmodules.segmentation.visualization.utils import (
    mask2contour,
    prediction_probs_to_cids,
    show_semantic_mask
)


def show_general(
        img_name: str,
        ant_path: str,
        save_path: None | str = None,
    ):
    """
    Args:
        img_name (str): image name
        ant_path (str): path to general.json
        save_path (None | str): save path. if None, not saved. 
    """
    general = json.load(open(ant_path, 'r'))
    data_dict = next(data_dict for data_dict in general["data"] if os.path.basename(data_dict["img_path"])==os.path.basename(img_name))
    
    img = cv2.imread(data_dict["img_path"])
    gt_contour_npy = np.load(data_dict["gt_contour_path"], allow_pickle=True)
    if "pd_filled_path" in data_dict:
        pd_filled_probs_npy = np.load(data_dict["pd_filled_path"], allow_pickle=True)
        pd_filled_cids_npy = prediction_probs_to_cids(
            pd_filled_probs_npy,
            data_dict.get("pd_boxes", None),
            data_dict.get("pd_probs", None)
        )
        pd_contour_npy = mask2contour(pd_filled_cids_npy)
    else:
        pd_contour_npy = None
    
    show_semantic_mask(
        general["categories"],
        img,
        gt_contour_npy,
        pd_contour_npy,
        save_path
    )


def show_coco(
        img_name: str,
        img_folder: str,
        ant_path: str,
        save_folder: str = ".tmp",
        use_cache: bool = True
    ):
    """
    Show an image with its ground truth in coco format.
    This func will convert coco format data into `general` format.
    Args:
        img_name (str): name of target image to be shown
        img_folder (str): path to the image folder
        ant_path (str): path to the coco label
        save_folder (None | str): folder saves the conversion result and visualized output
        use_cache (bool, optional): if true, the conversion execute once only.
    """
    general_path = os.path.join(save_folder, "general.json")
    
    if not use_cache or not os.path.exists(general_path):
        module = importlib.import_module(".format_conversion", package=__package__)
        module.coco2general(img_folder, ant_path, save_folder)
    save_path = os.path.join(save_folder, "vis_" + img_name)
    
    show_general(img_name, general_path, save_path)