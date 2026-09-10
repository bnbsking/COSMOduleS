from cosmodules.classification.active_learning import ClassificationActiveLearning


class TestClassificationActiveLearning:
    def test_run_single_label(self):
        ClassificationActiveLearning(
            pred_path = "/app/tests/integration/input/classification/prediction/single_label.json",
            save_path = "/app/tests/integration/output/classification/active_learning/single_label.json",
            loss_name = "entropy"
        )

    def test_run_single_label_background(self):
        ClassificationActiveLearning(
            pred_path = "/app/tests/integration/input/classification/prediction/single_label_background.json",
            save_path = "/app/tests/integration/output/classification/active_learning/single_label_background.json",
            loss_name = "entropy"
        )

    def test_run_multi_label(self):
        ClassificationActiveLearning(
            pred_path = "/app/tests/integration/input/classification/prediction/multi_label.json",
            save_path = "/app/tests/integration/output/classification/active_learning/multi_label.json",
            loss_name = "entropy"
        )


if __name__ == "__main__":
    obj = TestClassificationActiveLearning()
    obj.test_run_single_label()
    obj.test_run_single_label_background()
    obj.test_run_multi_label()
