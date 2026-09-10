import json
import os
import re
import shutil
from tempfile import TemporaryDirectory
from tqdm import tqdm
from typing import Dict, List, Literal

import cv2

from .components import VOCComponents
from .box_conversion import ConvertBoxFromAnyToVOC, ConvertBoxFromVOCToAny


class ConvertDatasetFromAnyToGeneral:
    @staticmethod
    def voc2general(
            img_path_list: List[str],
            ant_path_list: List[str],
            class_list: List[str],
            save_path: str | None = None,
        ) -> Dict:
        assert len(img_path_list) == len(ant_path_list)

        # initialization
        out = {"categories": class_list, "data": []}
        class_list.insert(0, "__background__") if class_list[0] != "__background__" else None

        for img_path, ant_path in tqdm(zip(img_path_list, ant_path_list)):
            # extract
            with open(ant_path, "r", encoding="utf-8") as f:
                xml = f.read()
            img_width  = int(re.findall("<width>([0-9]*)</width>", xml)[0])
            img_height = int(re.findall("<height>([0-9]*)</height>", xml)[0])
            class_name_list = re.findall("<name>(.*)</name>", xml)
            gt_cls = [class_list.index(class_name) for class_name in class_name_list]
            xmin_list = re.findall("<xmin>(.*)</xmin>", xml)
            ymin_list = re.findall("<ymin>(.*)</ymin>", xml)
            xmax_list = re.findall("<xmax>(.*)</xmax>", xml)
            ymax_list = re.findall("<ymax>(.*)</ymax>", xml)
            gt_boxes = [
                [int(xmin), int(ymin), int(xmax), int(ymax)]
                for xmin, ymin, xmax, ymax in zip(xmin_list, ymin_list, xmax_list, ymax_list)
            ]

            # collect
            out["data"].append(
                {
                    "img_path": os.path.abspath(img_path),
                    "img_width": img_width,
                    "img_height": img_height,
                    "gt_boxes": gt_boxes,
                    "gt_cls": gt_cls,
                    "pd_boxes": [],
                    "pd_probs": [],
                }
            )

        # save
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=4)
        return out

    @staticmethod
    def coco2general(
            img_folder: str,
            ant_path: str,
            save_path: str | None = None,
            cat_index_start: int = 1
        ) -> Dict:
        # initialization
        with open(ant_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
            class_list = ["__background__"] * (max(int(cat_dict["id"]) for cat_dict in coco['categories']) + 1)
            for cat_dict in coco['categories']:
                class_list[cat_dict["id"]] = cat_dict["name"]
        out = {"categories": class_list, "data": []}

        # Get image_id to all info
        img_id_to_all = {}
        for img_dict in coco['images']:
            img_path = os.path.join(img_folder, img_dict['file_name'])
            img_id_to_all[img_dict['id']] = {
                "img_path": os.path.abspath(img_path),
                "img_width": img_dict['width'],
                "img_height": img_dict['height'],
                "gt_boxes": [],
                "gt_cls": []
            }

        # Collect annotation
        for ant_dict in coco['annotations']:
            img_id = ant_dict['image_id']
            xmin, ymin, w, h = ant_dict['bbox']
            xmin, ymin, xmax, ymax = ConvertBoxFromAnyToVOC().run(
                src_type="coco",
                b1=xmin,
                b2=ymin,
                b3=w,
                b4=h
            )
            img_id_to_all[img_id]["gt_boxes"].append(
                    [int(xmin), int(ymin), int(xmax), int(ymax)]
                )
            img_id_to_all[img_id]["gt_cls"].append(
                int(ant_dict['category_id']) - cat_index_start + 1
            )
        out["data"] = list(img_id_to_all.values())

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=4)
        return out

    @staticmethod
    def yolo2general(
            img_path_list: List[str],
            ant_path_list: List[str],
            class_list: List[str],
            save_path: str | None = None,
            cat_index_start: int = 0
        ) -> Dict:
        assert len(img_path_list) == len(ant_path_list)
        
        # initialization
        out = {"categories": class_list, "data": []}
        class_list.insert(0, "__background__") if class_list[0] != "__background__" else None

        for img_path, ant_path in tqdm(zip(img_path_list, ant_path_list)):
            # extract
            img_height, img_width, _ = cv2.imread(img_path).shape
            with open(ant_path, "r", encoding="utf-8") as f:
                txt_lines = f.readlines()
            gt_boxes = []
            gt_cls = []
            for txt_line in txt_lines:
                if not txt_line.strip():
                    continue
                cid, cx, cy, w, h = txt_line.split(" ")
                xmin, ymin, xmax, ymax = ConvertBoxFromAnyToVOC().run(
                    src_type="yolo",
                    b1=float(cx),
                    b2=float(cy),
                    b3=float(w),
                    b4=float(h),
                    img_width=img_width,
                    img_height=img_height
                )
                gt_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                gt_cls.append(int(cid) - cat_index_start + 1)

            # collect
            out["data"].append(
                {
                    "img_path": os.path.abspath(img_path),
                    "img_width": img_width,
                    "img_height": img_height,
                    "gt_boxes": gt_boxes,
                    "gt_cls": gt_cls,
                    "pd_boxes": [],
                    "pd_probs": [],
                }
            )

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=4)
        return out


class ConvertDatasetFromGeneral2Any:
    @staticmethod
    def general2voc(ant_path: str, save_folder: str):
        # initialization
        with open(ant_path, "r", encoding="utf-8") as f:
            general = json.load(f)
            categories = general["categories"]
            data = general["data"]
        
        vocc = VOCComponents
        os.makedirs(save_folder, exist_ok=True)
        for data_dict in data:
            # collect
            filename = os.path.basename(data_dict["img_path"])
            out = vocc.get_xml(
                filename,
                data_dict["img_path"],
                data_dict["img_width"],
                data_dict["img_height"]
            )
            for (xmin, ymin, xmax, ymax), gt_cls in zip(data_dict["gt_boxes"], data_dict["gt_cls"]):
                out += vocc.get_obj(categories[gt_cls], xmin, ymin, xmax, ymax)
            out += vocc.get_end()
            
            # save
            save_path = os.path.join(save_folder, f"{filename.split('.')[0]}.xml")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(out)
            shutil.copy(data_dict["img_path"], save_folder)

    @staticmethod
    def general2yolo(ant_path: str, save_folder: str, cat_index_start: int = 0):
        # initialization
        with open(ant_path, "r", encoding="utf-8") as f:
            general = json.load(f)
            categories = general["categories"]
            data = general["data"]

        pad = lambda s: str(s) + '0'*(8-len(str(s)))
        os.makedirs(save_folder, exist_ok=True)
        for data_dict in data:
            # collect
            filename = os.path.basename(data_dict["img_path"])
            width = data_dict["img_width"]
            height = data_dict["img_height"]
            out = ""
            for (xmin, ymin, xmax, ymax), gt_cls in zip(data_dict["gt_boxes"], data_dict["gt_cls"]):
                cx, cy, w, h = ConvertBoxFromVOCToAny().run(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    target_type="yolo",
                    img_width=width,
                    img_height=height
                )
                out += f"{gt_cls - 1 + cat_index_start} {pad(cx)} {pad(cy)} {pad(w)} {pad(h)}\n"

            # save
            save_path = os.path.join(save_folder, f"{filename.split('.')[0]}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(out)
            shutil.copy(data_dict["img_path"], save_folder)
        
        with open(os.path.join(save_folder, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(categories[1:]))

    @staticmethod
    def general2coco(ant_path: str, save_folder: str, cat_index_start: int = 1):
        # initialization
        with open(ant_path, "r", encoding="utf-8") as f:
            general = json.load(f)
            categories = general["categories"]
            data = general["data"]
        out = {
            "images":[],
            "annotations":[],
            "categories": [
                {
                    "supercategory": "none",
                    "id": i,
                    "name": class_name
                } for i, class_name in enumerate(categories, cat_index_start)
            ]
        }
        
        os.makedirs(save_folder, exist_ok=True)
        total_labels = 0
        for img_id, data_dict in enumerate(data):
            out["images"].append(
                {
                    "file_name": os.path.basename(data_dict["img_path"]),
                    "width": data_dict["img_width"],
                    "height": data_dict["img_height"],
                    "id": img_id
                }
            )
            for (xmin, ymin, xmax, ymax), gt_cls in zip(data_dict["gt_boxes"], data_dict["gt_cls"]):
                xmin, ymin, w, h = ConvertBoxFromVOCToAny().run(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    target_type="coco"
                )
                out["annotations"].append(
                    {
                        "area": w * h,
                        "iscrowd": 0,
                        "bbox": [xmin, ymin, w, h],
                        "category_id": gt_cls - 1 + cat_index_start,
                        "ignore": 0,
                        "segmentation": [],
                        "image_id": img_id,
                        "id": total_labels
                    }
                )
                total_labels += 1
            shutil.copy(data_dict["img_path"], save_folder)

        save_path = os.path.join(save_folder, "coco.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)


class ConvertDatasetFromAnyToAny:
    @staticmethod
    def voc2any(
            tgt_format: Literal["voc", "yolo", "coco"],
            img_path_list: List[str],
            ant_path_list: List[str],
            class_list: List[str],
            save_folder: str,
        ):
        with TemporaryDirectory() as tmp_dir:
            general_path = os.path.join(tmp_dir, "general.json") 
            ConvertDatasetFromAnyToGeneral.voc2general(
                img_path_list = img_path_list,
                ant_path_list = ant_path_list,
                class_list = class_list,
                save_path = general_path
            )
            getattr(ConvertDatasetFromGeneral2Any, f"general2{tgt_format}")(
                ant_path = general_path,
                save_folder = save_folder
            )
    
    @staticmethod
    def yolo2any(
            tgt_format: Literal["voc", "yolo", "coco"],
            img_path_list: List[str],
            ant_path_list: List[str],
            class_list: List[str],
            save_folder: str,
        ):
        with TemporaryDirectory() as tmp_dir:
            general_path = os.path.join(tmp_dir, "general.json")
            ConvertDatasetFromAnyToGeneral.yolo2general(
                img_path_list = img_path_list,
                ant_path_list = ant_path_list,
                class_list = class_list,
                save_path = general_path
            )
            getattr(ConvertDatasetFromGeneral2Any, f"general2{tgt_format}")(
                ant_path = general_path,
                save_folder = save_folder
            )
    
    @staticmethod
    def coco2any(
            tgt_format: Literal["voc", "yolo", "coco"],
            img_folder: str,
            ant_path: str,
            save_folder: str,
        ):
        with TemporaryDirectory() as tmp_dir:
            general_path = os.path.join(tmp_dir, "general.json")
            ConvertDatasetFromAnyToGeneral.coco2general(
                img_folder = img_folder,
                ant_path = ant_path,
                save_path = general_path
            )
            getattr(ConvertDatasetFromGeneral2Any, f"general2{tgt_format}")(
                ant_path = general_path,
                save_folder = save_folder
            )
