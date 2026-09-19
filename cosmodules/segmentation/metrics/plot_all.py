import os
from typing import Dict, List

import numpy as np

from cosmodules.utils.plot import (
    plot_bar,
    plot_pr_curves,
    plot_prf_curves,
    plot_confusion
)


def plot_all(
        class_name_list: List[str],
        ap_list: List[float],
        refined_pr_curves: List[Dict[str, List[float]]],
        pr_curves: List[Dict[str, List[float]]],
        confusion: np.ndarray,
        confusion_col_norm: np.ndarray,
        confusion_row_norm: np.ndarray,
        save_folder: str,
    ):
    save_path = os.path.join(save_folder, "ap_list.jpg")
    plot_bar(class_name_list, ap_list, save_path)

    save_path = os.path.join(save_folder, "refined_pr_curves.jpg")
    plot_pr_curves(class_name_list, refined_pr_curves, save_path)

    save_path = os.path.join(save_folder, "prf_curves.jpg")
    plot_prf_curves(class_name_list, pr_curves, save_path)

    save_path = os.path.join(save_folder, "confusion.jpg")
    plot_confusion(class_name_list, confusion, confusion_col_norm, confusion_row_norm, save_path)
