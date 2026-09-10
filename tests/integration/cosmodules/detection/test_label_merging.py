from cosmodules.detection.label_merging import DetectionLabelMerging


class TestDetectionLabelMerging:
    def test_run(self):
        cfg_path_list = [
            "/app/tests/integration/input/detection/data/general/general.json",
            "/app/tests/integration/input/detection/data/general_another_labeler/general.json",
        ]
        save_path = "tests/integration/output/detection/label_merging/general.json"
        ties_handling = "union"

        DetectionLabelMerging(cfg_path_list, save_path, ties_handling)


if __name__ == "__main__":
    test = TestDetectionLabelMerging()
    test.test_run()
    