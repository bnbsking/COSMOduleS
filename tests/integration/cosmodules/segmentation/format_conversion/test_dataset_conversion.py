from cosmodules.segmentation.format_conversion.dataset_conversion import (
    get_category_list,
    coco2general
)


def test_get_category_list():
    categories = [
        {"id": 1, "name": "cat"},
        {"id": 2, "name": "dog"}
    ]
    category_list = get_category_list(categories)
    assert category_list == ["__background__", "cat", "dog"]


def test_coco2general():
    img_folder = "/app/tests/integration/input/segmentation/data/coco"
    ant_path = "/app/tests/integration/input/segmentation/data/coco/coco.json"
    save_folder = "/app/tests/integration/output/segmentation/format_conversion/coco2general"
    coco2general(img_folder, ant_path, save_folder)


if __name__ == "__main__":
    test_get_category_list()
    test_coco2general()
