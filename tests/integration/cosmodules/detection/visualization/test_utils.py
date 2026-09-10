import cv2

from cosmodules.detection.visualization.utils import (
    get_digit_patch,
    get_confidence_patch,
    DetectionLabel,
    show
)


def test_get_digit_patch():
    save_path = "/app/tests/integration/output/detection/visualization/digit_patch.jpg"

    digit_patch = get_digit_patch(digit=5)

    assert digit_patch.shape == (30, 20, 3)
    cv2.imwrite(save_path, (digit_patch * 255).astype("uint8"))


def test_get_confidence_patch():
    save_path = "/app/tests/integration/output/detection/visualization/confidence_patch.jpg"

    confidence_patch = get_confidence_patch(unit_digit=5, tens_digit=3)

    assert confidence_patch.shape == (30, 60, 3)
    cv2.imwrite(save_path, (confidence_patch * 255).astype("uint8"))


def test_show():
    class_list = ["__background__", "class1", "class2"]
    label = DetectionLabel(
        img_path="/app/tests/integration/input/detection/data/general/pic1.jpg",
        gt_boxes=[(8, 15, 715, 630), (733, 64, 1174, 588)],
        gt_cls=[1, 2],
        pd_boxes=[(8, 15, 715, 630), (733, 64, 1174, 588)],
        pd_probs=[[0.0, 0.9, 0.1], [0.2, 0.3, 0.5]]
    )
    save_path = "/app/tests/integration/output/detection/visualization/show.jpg"

    show(class_list=class_list, label=label, save_path=save_path)


if __name__ == "__main__":
    test_get_digit_patch()
    test_get_confidence_patch()
    test_show()