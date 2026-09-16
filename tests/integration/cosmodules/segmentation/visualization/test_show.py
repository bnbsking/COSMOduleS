from cosmodules.segmentation.visualization.show import (
    show_general,
    show_coco
)


def test_show_general():
    show_general(
        img_name="img1.jpg",
        ant_path="tests/integration/input/segmentation/data/general/general.json",
        save_path="tests/integration/output/segmentation/visualization/show_general.jpg"
    )


if __name__ == "__main__":
    test_show_general()