import cv2
import numpy as np

from cosmodules.segmentation.visualization.utils import (
    show_semantic_mask
)


def test_show_semantic_mask():
    categories = [
        "__background__",
        "rectangle",
        "triangle"
    ]
    img = cv2.imread("/app/tests/integration/input/segmentation/data/general/img1.jpg")
    gt_contour_npy = np.load("/app/tests/integration/input/segmentation/data/general/gt_contour_img1.npy", allow_pickle=True)
    pd_contour_npy = None
    save_path = "/app/tests/integration/output/segmentation/visualization/show_semantic_mask.jpg"

    show_semantic_mask(
        categories,
        img,
        gt_contour_npy,
        pd_contour_npy,
        save_path
    )


if __name__ == "__main__":
    test_show_semantic_mask()
