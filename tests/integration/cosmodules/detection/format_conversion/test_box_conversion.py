from cosmodules.detection.format_conversion.box_conversion import (
    ConvertBoxFromAnyToVOC,
    ConvertBoxFromVOCToAny
)


class TestConvertBoxFromAnyToVOC:
    def test_voc(self):
        converter = ConvertBoxFromAnyToVOC()
        xmin, ymin, xmax, ymax = converter.run(
            src_type="voc",
            b1=100,
            b2=150,
            b3=400,
            b4=350
        )
        assert (xmin, ymin, xmax, ymax) == (100, 150, 400, 350)

    def test_coco(self):
        converter = ConvertBoxFromAnyToVOC()
        xmin, ymin, xmax, ymax = converter.run(
            src_type="coco",
            b1=100,
            b2=150,
            b3=300,
            b4=200
        )
        assert (xmin, ymin, xmax, ymax) == (100, 150, 400, 350)

    def test_yolo(self):
        converter = ConvertBoxFromAnyToVOC()
        xmin, ymin, xmax, ymax = converter.run(
            src_type="yolo",
            b1=0.25,
            b2=0.333333,
            b3=0.375,
            b4=0.333333,
            img_width=800,
            img_height=600
        )
        assert (xmin, ymin, xmax, ymax) == (50, 100, 350, 300)


class TestConvertBoxFromVOCToAny:
    def test_voc(self):
        converter = ConvertBoxFromVOCToAny()
        xmin, ymin, xmax, ymax = converter.run(
            xmin=100,
            ymin=150,
            xmax=400,
            ymax=350,
            target_type="voc"
        )
        assert (xmin, ymin, xmax, ymax) == (100, 150, 400, 350)

    def test_coco(self):
        converter = ConvertBoxFromVOCToAny()
        xmin, ymin, w, h = converter.run(
            xmin=100,
            ymin=150,
            xmax=400,
            ymax=350,
            target_type="coco"
        )
        assert (xmin, ymin, w, h) == (100, 150, 300, 200)

    def test_yolo(self):
        converter = ConvertBoxFromVOCToAny()
        cx, cy, w, h = converter.run(
            xmin=100,
            ymin=150,
            xmax=400,
            ymax=350,
            target_type="yolo",
            img_width=800,
            img_height=600
        )
        assert (cx, cy, w, h) == (0.3125, 0.416667, 0.375, 0.333333)


if __name__ == "__main__":
    obj = TestConvertBoxFromAnyToVOC()
    obj.test_voc()
    obj.test_coco()
    obj.test_yolo()

    obj = TestConvertBoxFromVOCToAny()
    obj.test_voc()
    obj.test_coco()
    obj.test_yolo()