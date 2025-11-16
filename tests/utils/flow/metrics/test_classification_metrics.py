import numpy as np

from cosmodules.utils.flow.metrics.classification_metrics import ClassificationMetricsFlow


class TestClassificationMetricsFlow:
    def test_run_single_label(self):
        obj = ClassificationMetricsFlow(
            num_classes = 2,
            labels = np.array([0, 0, 1]),
            predictions = np.array([
                [0.95, 0.05],
                [0.1, 0.9],
                [0.2, 0.8]
            ]),
            save_path = "/app/example/classification/output/metrics_new/single_label/metrics.json",
        )
        results = obj.run()
        print("Results:", results)

    def test_run_multi_label(self):
        obj = ClassificationMetricsFlow(
            num_classes = 2,
            labels = np.array([
                [1, 0],
                [1, 1],
                [1, 1]
            ]),
            predictions = np.array([
                [[0.05, 0.95], [0.8, 0.2]],
                [[0.7, 0.3], [0.15, 0.85]],
                [[0.4, 0.6], [0.25, 0.75]]
            ]),
            save_path = "/app/example/classification/output/metrics_new/multi_label/metrics.json",
        )
        results = obj.run()
        print("Results:", results)

    def test_run_single_label_background(self):
        obj = ClassificationMetricsFlow(
            num_classes = 2,
            labels = np.array([1, 1, 1]),
            predictions = np.array([
                [0.95, 0.05],
                [0.1, 0.9],
                [0.2, 0.8]
            ]),
            save_path = "/app/example/classification/output/metrics_new/single_label_background/metrics.json",
            start_idx = 1,
        )
        results = obj.run()
        print("Results:", results)


if __name__ == "__main__":
    test = TestClassificationMetricsFlow()
    test.test_run_single_label()
    test.test_run_multi_label()
    test.test_run_single_label_background()
