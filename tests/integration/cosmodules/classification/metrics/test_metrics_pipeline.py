import json

import numpy as np

from cosmodules.classification.metrics.metrics_pipeline import ClassificationMetricsPipeline


class TestClassificationMetricsPipeline:
    def test_run_single_label(self):
        num_classes = 2
        labels = np.array([0, 0, 1])
        predictions = np.array([
            [0.95, 0.05],
            [0.1, 0.9],
            [0.2, 0.8]
        ])
        start_idx = 0

        pipeline = ClassificationMetricsPipeline()
        out = pipeline.run(num_classes, labels, predictions, start_idx)
        print(out)
        
        with open("tests/integration/output/classification/metrics/single_label_metrics.json", "w") as f:
            json.dump(out, f, indent=4)

    def test_run_single_label_background(self):
        num_classes = 2
        labels = np.array([1, 1, 1])
        predictions = np.array([
            [0.95, 0.05],
            [0.1, 0.9],
            [0.2, 0.8]
        ])
        start_idx = 1

        pipeline = ClassificationMetricsPipeline()
        out = pipeline.run(num_classes, labels, predictions, start_idx)
        print(out)

        with open("tests/integration/output/classification/metrics/single_label_background_metrics.json", "w") as f:
            json.dump(out, f, indent=4)

    def test_run_multi_label(self):
        num_classes = 2
        labels = np.array([
            [1, 0],
            [1, 1],
            [1, 1]
        ])
        predictions = np.array([
            [
                [0.05, 0.95],
                [0.8, 0.2]
            ],
            [
                [0.7, 0.3],
                [0.15, 0.85]
            ],
            [
                [0.4, 0.6],
                [0.25, 0.75]
            ]
        ])  # (samples, classes, 2)
        start_idx = 0  # always 0

        pipeline = ClassificationMetricsPipeline()
        out = pipeline.run(num_classes, labels, predictions, start_idx)
        print(out)

        with open("tests/integration/output/classification/metrics/multi_label_metrics.json", "w") as f:
            json.dump(out, f, indent=4)


if __name__ == "__main__":
    test_pipeline = TestClassificationMetricsPipeline()
    test_pipeline.test_run_single_label()
    test_pipeline.test_run_single_label_background()
    test_pipeline.test_run_multi_label()