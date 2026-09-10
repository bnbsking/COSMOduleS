import glob

from cosmodules.detection.visualization.show import (
    show_general,
    show_coco,
    show_voc,
    show_yolo
)


def test_show_general():
    img_path = "/app/tests/integration/input/detection/data/general/pic1.jpg"
    ant_path = "/app/tests/integration/input/detection/data/general/general.json"
    save_path = "/app/tests/integration/output/detection/visualization/show_general.jpg"
    show_general(img_path, ant_path, save_path)


def test_show_coco():
    img_path = "/app/tests/integration/input/detection/data/coco/pic1.jpg"
    ant_path = "/app/tests/integration/input/detection/data/coco/coco.json"
    save_path = "/app/tests/integration/output/detection/visualization/show_coco.jpg"
    show_coco(img_path, ant_path, save_path)


def test_show_voc():
    img_name = "pic1.jpg"
    img_path_list = sorted(glob.glob("/app/tests/integration/input/detection/data/voc/*.jpg"))
    ant_path_list = sorted(glob.glob("/app/tests/integration/input/detection/data/voc/*.xml"))
    class_list = ["dog", "cat"]
    save_path = "/app/tests/integration/output/detection/visualization/show_voc.jpg"
    show_voc(img_name, img_path_list, ant_path_list, class_list, save_path)


def test_show_yolo():
    img_name = "pic1.jpg"
    img_path_list = sorted(glob.glob("/app/tests/integration/input/detection/data/yolo/*.jpg"))
    ant_path_list = [p for p in sorted(glob.glob("/app/tests/integration/input/detection/data/yolo/*.txt")) if not p.endswith("classes.txt")]
    class_list = ["dog", "cat"]
    save_path = "/app/tests/integration/output/detection/visualization/show_yolo.jpg"
    show_yolo(img_name, img_path_list, ant_path_list, class_list, save_path)


if __name__ == "__main__":
    test_show_general()
    test_show_coco()
    test_show_voc()
    test_show_yolo()