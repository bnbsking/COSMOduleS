import json

import numpy as np

from cosmodules.detection.metrics.metrics_pipeline import DetectionMetricsPipeline


class TestDetectionMetricsPipeline:
    def test_run(self):
        num_classes = 3
        labels = [
            np.array([
                [1, 0, 23, 220, 228]
            ]),
            np.array([
                [1, 8, 15, 715, 630],
                [2, 733, 64, 1174, 588]
            ]),
            np.array([
                [1, 35, 7, 112, 114],
                [1, 171, 58, 263, 121],
                [2, 114, 39, 170, 118],
                [2, 259, 62, 326, 118]
            ])
        ]  # class_id, x_min, y_min, x_max, y_max
        detections = [
            np.array([
                [0, 23, 220, 228, 0.95, 1],
                [30, 90, 80, 150, 0.8, 2]
            ]),
            np.array([
                [8, 15, 715, 630, 0.85, 1],
                [733, 64, 1174, 588, 0.6, 1]
            ]),
            np.array([
                [35, 7, 112, 114, 0.75, 1],
                [171, 58, 211, 88, 0.45, 1],
                [114, 39, 170, 118, 0.99, 2]
            ])
        ] # xmin, ymin, xmax, ymax, confidence, class_id

        pipeline = DetectionMetricsPipeline()
        out = pipeline.run(num_classes, labels, detections)
        print(out)

        with open("tests/integration/output/detection/metrics/metrics.json", "w") as f:
            json.dump(out, f, indent=4)
        

if __name__ == "__main__":
    obj = TestDetectionMetricsPipeline()
    obj.test_run()