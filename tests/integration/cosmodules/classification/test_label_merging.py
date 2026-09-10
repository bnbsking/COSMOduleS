from cosmodules.classification.label_merging import ClassificationLabelMerging


class TestClassificationLabelMerging:
    def test_run_single_label(self):
        ClassificationLabelMerging(
            cfg_path_list = [
                "/app/tests/integration/input/classification/data/single_label.json",
                "/app/tests/integration/input/classification/data_another_labeler/single_label.json"
            ],
            save_path = "/app/tests/integration/output/classification/label_merging/single_label.json"
        )

    def test_run_single_label_background(self):
        ClassificationLabelMerging(
            cfg_path_list = [
                "/app/tests/integration/input/classification/data/single_label_background.json",
                "/app/tests/integration/input/classification/data_another_labeler/single_label_background.json",
            ],
            save_path = "/app/tests/integration/output/classification/label_merging/single_label_background.json"
        )

    def test_run_multi_label(self):
        ClassificationLabelMerging(
            cfg_path_list = [
                "/app/tests/integration/input/classification/data/multi_label.json",
                "/app/tests/integration/input/classification/data_another_labeler/multi_label.json",
            ],
            save_path = "/app/tests/integration/output/classification/label_merging/multi_label.json"
        )


if __name__ == "__main__":
    obj = TestClassificationLabelMerging()
    obj.test_run_single_label()
    obj.test_run_single_label_background()
    obj.test_run_multi_label()
