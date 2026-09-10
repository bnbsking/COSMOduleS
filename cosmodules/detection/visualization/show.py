import json
import os
from tempfile import TemporaryDirectory
from typing import List

from cosmodules.detection.format_conversion.dataset_conversion import (
    ConvertDatasetFromAnyToGeneral
)
from cosmodules.detection.visualization.utils import DetectionLabel, show


def show_general(img_path: str, ant_path: str, save_path: str | None = None):
    with open(ant_path, "r", encoding="utf-8") as f:
        general = json.load(f)
    data_dict = next(data_dict for data_dict in general["data"] \
        if os.path.basename(data_dict["img_path"]) == os.path.basename(img_path)
    )

    class_list = general["categories"]
    label = DetectionLabel(
        img_path=img_path,
        gt_boxes=[tuple(tup) for tup in data_dict["gt_boxes"]],
        gt_cls=data_dict["gt_cls"],
        pd_boxes=[tuple(tup) for tup in data_dict.get("pd_boxes", [])],
        pd_probs=data_dict.get("pd_probs", []),
    )
    show(class_list, label, save_path)


def show_coco(
        img_path: str,
        ant_path: str,
        save_path: str | None = None
    ):
    """Massive call is inefficient, convert to general and use show_general instead"""
    with TemporaryDirectory() as tmpdir:
        general_path = os.path.join(tmpdir, "general.json")
        ConvertDatasetFromAnyToGeneral().coco2general(
            img_folder=os.path.dirname(img_path),
            ant_path=ant_path,
            save_path=general_path,
        )
        show_general(img_path, general_path, save_path)


def show_voc(
        img_name: str,
        img_path_list: List[str],
        ant_path_list: List[str],
        class_list: List[str],
        save_path: str | None = None
    ):
    """Massive call is inefficient, convert to general and use show_general instead"""
    with TemporaryDirectory() as tmpdir:
        general_path = os.path.join(tmpdir, "general.json")
        ConvertDatasetFromAnyToGeneral().voc2general(
            img_path_list=img_path_list,
            ant_path_list=ant_path_list,
            class_list=class_list,
            save_path=general_path,
        )
        img_path = next(path for path in img_path_list if os.path.basename(path) == img_name)
        show_general(img_path, general_path, save_path)


def show_yolo(
        img_name: str,
        img_path_list: List[str],
        ant_path_list: List[str],
        class_list: List[str],
        save_path: str | None = None
    ):
    """Massive call is inefficient, convert to general and use show_general instead"""
    with TemporaryDirectory() as tmpdir:
        general_path = os.path.join(tmpdir, "general.json")
        ConvertDatasetFromAnyToGeneral().yolo2general(
            img_path_list=img_path_list,
            ant_path_list=ant_path_list,
            class_list=class_list,
            save_path=general_path,
        )
        img_path = next(path for path in img_path_list if os.path.basename(path) == img_name)
        show_general(img_path, general_path, save_path)
