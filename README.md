# COSMOduleS: Classification, Object detection, Segmentation MOduleS


Comprehensive preprocessing and post-processing tools for common **Computer Vision** tasks.

("bg" means containing background classes which the index must start from 0 in this repo.)


| Tasks | Subtasks | Defined<br>Format | Visualization | Format<br>Conversion | Metrics | Label<br>Merging | Active<br>Learning |
| - | - | - | - | - | - | - | - |
| Classification | single-label<br> single-label-bg<br> multi-label(binary)<br> | [single_label](tests/integration/input/classification/data/single_label.json)<br> [single_label_bg](tests/integration/input/classification/data/single_label_background.json)<br> [multi_label](tests/integration/input/classification/data/multi_label.json) | - | - | [ALL](tests/integration/cosmodules/classification/metrics) | [ALL](tests/integration/cosmodules/classification/test_label_merging.py) | [Entropy](tests/integration/cosmodules/classification/test_active_learning.py) |
| Detection      | - | [coco](tests/integration/input/detection/data/coco/coco.json)<br> [voc](tests/integration/input/detection/data/voc)<br> [yolo](tests/integration/input/detection/data/yolo)<br> [**GENERAL**](tests/integration/input/detection/data/general)<br> | [ALL](tests/integration/cosmodules/detection/visualization/test_show.py) | [between ANY<br>two types](tests/integration/cosmodules/detection/format_conversion/test_dataset_conversion.py) | [ALL](tests/integration/cosmodules/detection/metrics) | [ALL](tests/integration/cosmodules/detection/test_label_merging.py) | [horizontal<br>flip](tests/integration/cosmodules/detection/test_active_learning.py) |
| Segmentation   | instance<br> semantic<br> | [coco](tests/integration/input/segmentation/data/coco)<br> [**GENERAL**](tests/integration/input/segmentation/data/general) | [ALL](tests/integration/cosmodules/segmentation/visualization/test_show.py) | [coco2general](tests/integration/cosmodules/segmentation/format_conversion/test_dataset_conversion.py) | [ALL](tests/integration/cosmodules/segmentation/metrics) | - | [instance<br>semantic<br>](.) |


(segmentation not refactored yet)


## Quick start

+ SDK
    ```bash
    pip install -e .
    ```

+ Debug
    ```bash
    docker compose build
    docker compose up -d
    ```


## Motivation

+ Classification

| task                       | data format             | compute class-0 metrics | threshold optimization |  
| -                          | -                       | -                       | -                      |
| multi-class classification | single label            | V                       |                        |
| anomaly classification     | single label background |                         | V (fore-back)          |
| multi-label classification | multi_label             | V                       | V (mean)               |

+ Object Detection
    + Develop a **GENERAL** format to be the most convenient.
    + The formats can be summarized as following:

| format | extension | files     | type  | box                      | disadvantage |
| -      | -         | -         | -     | -                        | -            |
| coco   | .json     | 1         | int   | (xmin, ymin, w, h)       | get label of an image |
| yolo   | .txt      | len(imgs) | float | (cx, cy, w/2, h/2)       | visualization, compute metrics, etc. |
| voc    | .xml      | len(imgs) | int   | (xmin, ymin, xmax, ymax) | get class list |
| general| .json | 1 | int | (xmin, ymin, xmax, ymax) | **NO** |


+ Segmentation
    + Develop a **GENERAL** format to be the most convenient.
    + prediction format
        + numpy, shape=(num_classes, H, W)
        + value=0~1 


| Includes         | Content | Advantage |
| -                | -       | -         |
| general.json     | Includes every imgs: path, contour, filled and boxes with class | Searching | 
| gt_contour_*.npy | (H, W) with {0, 1, ..., num_classes} int | Plotting |
| gt_filled_*.npy  | (num_classes, H, W) with 0 or 1 int values | Compute IOU for Metrics |
| *.jpg            | Raw data | - |





## **Examples**
+ detection visualization
![.](pictures/detection_visualization.jpg)

+ confusion
![.](pictures/confusion.jpg)

+ prf curves
![.](pictures/prf_curves.jpg)


## **More**
+ Feel free to ask if you have any question.
+ Notice not supported
    + segmentation general2coco
    + segmentation label merging


## **Acknowledgement**
+ Confusion Matrix reference [here](https://github.com/kaanakan/object_detection_confusion_matrix/blob/master/confusion_matrix.py)
