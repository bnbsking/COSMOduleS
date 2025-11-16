import numpy as np

from cosmodules.utils.flow.metrics.detection_metrics import DetectionMetricsFlow


class TestDetectionMetricsFlow:
    def test_run(self):
        obj = DetectionMetricsFlow(
            num_classes = 3,
            labels = [
                np.array([[1, 0, 23, 220, 228]]),
                np.array([[1, 8, 15, 715, 630], [2, 733, 64, 1174, 588]]),
                np.array([[1, 35, 7, 112, 114], [1, 171, 58, 263, 121], [2, 114, 39, 170, 118], [2, 259, 62, 326, 118]]),
            ],
            predictions = [
                np.array([
                    [0, 23, 220, 228, 0.95, 1],
                    [30, 90, 80, 150, 0.8, 2],
                ]),
                np.array([
                    [8, 15, 715, 630, 0.85, 1],
                    [733, 64, 1174, 588, 0.6, 1],
                ]),
                np.array([
                    [35, 7, 112, 114, 0.75, 1],
                    [171, 58, 211, 88, 0.45, 1],
                    [114, 39, 170, 118, 0.99, 2],
                ]),
            ],
            save_path = "/app/example/detection/output/metrics_new/metrics.json",
        )
        results = obj.run()
        print("Results:", results)


if __name__ == "__main__":
    test = TestDetectionMetricsFlow()
    test.test_run()
