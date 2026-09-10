import glob

from cosmodules.detection.format_conversion.dataset_conversion import (
    ConvertDatasetFromAnyToGeneral,
    ConvertDatasetFromGeneral2Any,
    ConvertDatasetFromAnyToAny,
)


class TestConvertDatasetFromAnyToGeneral:
    def test_voc2general(self):
        converter = ConvertDatasetFromAnyToGeneral()
        out = converter.voc2general(
            img_path_list = sorted(glob.glob("tests/integration/input/detection/data/voc/*.jpg")),
            ant_path_list = sorted(glob.glob("tests/integration/input/detection/data/voc/*.xml")),
            class_list = ["dog", "cat"],
            save_path = "tests/integration/output/detection/format_conversion/voc2general.json"
        )
        assert isinstance(out, dict)

    def test_coco2general(self):
        converter = ConvertDatasetFromAnyToGeneral()
        out = converter.coco2general(
            img_folder = "tests/integration/input/detection/data/coco",
            ant_path = "tests/integration/input/detection/data/coco/coco.json",
            save_path = "tests/integration/output/detection/format_conversion/coco2general.json"
        )
        assert isinstance(out, dict)

    def test_yolo2general(self):
        converter = ConvertDatasetFromAnyToGeneral()
        out = converter.yolo2general(
            img_path_list = sorted(glob.glob("tests/integration/input/detection/data/yolo/*.jpg")),
            ant_path_list = [p for p in sorted(glob.glob("tests/integration/input/detection/data/yolo/*.txt")) if not p.endswith("classes.txt")],
            class_list = ["dog", "cat"],
            save_path = "tests/integration/output/detection/format_conversion/yolo2general.json"
        )
        assert isinstance(out, dict)
        

class TestConvertDatasetFromGeneral2Any:
    def test_general2voc(self):
        converter = ConvertDatasetFromGeneral2Any()
        converter.general2voc(
            ant_path = "tests/integration/input/detection/data/general/general.json",
            save_folder = "tests/integration/output/detection/format_conversion/general2voc"
        )

    def test_general2coco(self):
        converter = ConvertDatasetFromGeneral2Any()
        converter.general2coco(
            ant_path = "tests/integration/input/detection/data/general/general.json",
            save_folder = "tests/integration/output/detection/format_conversion/general2coco"
        )

    def test_general2yolo(self):
        converter = ConvertDatasetFromGeneral2Any()
        converter.general2yolo(
            ant_path = "tests/integration/input/detection/data/general/general.json",
            save_folder = "tests/integration/output/detection/format_conversion/general2yolo"
        )


class TestConvertDatasetFromAnyToAny:
    def test_yolo2coco(self):
        converter = ConvertDatasetFromAnyToAny()
        converter.yolo2any(
            tgt_format = "coco",
            img_path_list = sorted(glob.glob("tests/integration/input/detection/data/yolo/*.jpg")),
            ant_path_list = [p for p in sorted(glob.glob("tests/integration/input/detection/data/yolo/*.txt")) if not p.endswith("classes.txt")],
            class_list = ["dog", "cat"],
            save_folder = "tests/integration/output/detection/format_conversion/yolo2coco"
        )


if __name__ == "__main__":
    obj = TestConvertDatasetFromAnyToGeneral()
    obj.test_voc2general()
    obj.test_coco2general()
    obj.test_yolo2general()
    
    obj = TestConvertDatasetFromGeneral2Any()
    obj.test_general2voc()
    obj.test_general2coco()
    obj.test_general2yolo()

    obj = TestConvertDatasetFromAnyToAny()
    obj.test_yolo2coco()