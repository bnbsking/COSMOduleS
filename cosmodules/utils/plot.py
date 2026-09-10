import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_bar(
        class_name_list: List[str],
        value_list: List[float],
        save_path: str,
        title: str | None = None,
    ):
    assert len(class_name_list) == len(value_list)
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title(title, fontsize=16) if title is not None else None
    ax.bar(class_name_list, value_list)
    for i, value in enumerate(value_list):
        ax.text(i, value, value, ha="center", va="bottom", fontsize=16)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_pr_curves(
        class_name_list: List[str],
        refined_pr_curves: List[Dict[str, List[float]]],
        save_path: str
    ):
    assert len(class_name_list) == len(refined_pr_curves)
    num_classes = len(refined_pr_curves)
    plt.figure(figsize=(6 * num_classes, 4))
    for cid in range(num_classes):
        plt.subplot(1, num_classes, 1 + cid)
        plt.scatter(refined_pr_curves[cid]["recall"], refined_pr_curves[cid]["precision"])
        plt.plot(refined_pr_curves[cid]["recall"], refined_pr_curves[cid]["precision"])
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid('on')
        plt.title(f"{cid}-{class_name_list[cid]}", fontsize=16)
        plt.xlabel("recall", fontsize=16)
        plt.ylabel("precision", fontsize=16)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


def plot_prf_curves(
        class_name_list: List[str],
        pr_curves: List[Dict[str, List[float]]],
        save_path: str
    ):
    assert len(class_name_list) == len(pr_curves)
    num_classes = len(class_name_list)
    plt.figure(figsize=(6 * num_classes, 4))
    for cid in range(num_classes):
        f1_arr = [2 * p * r / (p + r + 1e-10) for p, r in \
            zip(pr_curves[cid]["precision"], pr_curves[cid]["recall"])]
        plt.subplot(1, num_classes, 1 + cid)
        plt.plot(pr_curves[cid]["precision"])
        plt.plot(pr_curves[cid]["recall"])
        plt.plot(f1_arr)
        plt.xlim(-5, 105)
        plt.ylim(-0.05, 1.05)
        plt.grid('on')
        plt.title(f"{cid}-{class_name_list[cid]}", fontsize=16)
        plt.xlabel("threshold", fontsize=16)
        plt.legend(labels=["precision", "recall", "f1"], fontsize=12)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


def plot_confusion(
        class_name_list: List[str],
        confusion: np.ndarray,
        confusion_col_norm: np.ndarray,
        confusion_row_norm: np.ndarray,
        save_path: str
    ):
    assert len(confusion) == len(confusion[0])
    assert len(confusion) == len(class_name_list) or len(confusion) == len(class_name_list) + 1
    assert confusion.shape == confusion_col_norm.shape == confusion_row_norm.shape

    num_classes = len(confusion)
    if num_classes == len(class_name_list):
        class_list = class_name_list
    else:
        class_list = ["BG"] + class_name_list
    matrix_plot_list = [confusion_col_norm, confusion_col_norm, confusion_row_norm]
    matrix_text_list = [confusion, confusion_col_norm, confusion_row_norm]
    title_list = ["confusion", "col norm (precision)", "row norm (recall)"]

    plt.figure(figsize=(15,5))
    
    for i, (matplt, mattxt, title) in enumerate(zip(matrix_plot_list, matrix_text_list, title_list)):
        fig = plt.subplot(1, 3, 1+i)
        plt.title(title, fontsize=12)
        plt.xlabel("PD", fontsize=12)
        plt.ylabel("GT", fontsize=12)
        fig.set_xticks(np.arange(num_classes)) # values
        fig.set_xticklabels(class_list)  # labels
        fig.set_yticks(np.arange(num_classes))  # values
        fig.set_yticklabels(class_list)  # labels
        plt.imshow(matplt, cmap=mpl.cm.Blues, interpolation='nearest', vmin=0, vmax=1)
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, round(mattxt[i][j], 2), ha="center", va="center", \
                    color="black" if matplt[i][j]<0.9 else "white", fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()