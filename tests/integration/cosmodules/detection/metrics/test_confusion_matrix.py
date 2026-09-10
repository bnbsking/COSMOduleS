from collections import Counter

import numpy as np

from cosmodules.detection.metrics.confusion_matrix import DetectionConfusionMatrix


class TestDetectionConfusionMatrix:
    def test_run(self):
        num_classes = 3
        detections = np.array([
            [8, 15, 715, 630, 0.85, 1],
            [733, 64, 1174, 588, 0.6, 1]
        ])  # xmin, ymin, xmax, ymax, confidence, class_id
        labels = np.array([
            [1, 8, 15, 715, 630],
            [2, 733, 64, 1174, 588]
        ])  # class_id, x_min, y_min, x_max, y_max
        img_idx = 7

        cm = DetectionConfusionMatrix(num_classes, img_idx=img_idx)
        cm.process_batch(detections, labels)

        confusion = cm.get_confusion().tolist()
        confusion_with_img_indices = cm.get_confusion_with_img_indices()
        assert confusion == [[0, 0, 0], [0, 1, 0], [0, 1, 0]]
        assert confusion_with_img_indices == [
            [Counter(), Counter(), Counter()],
            [Counter(), Counter({7: 1}), Counter()],
            [Counter(), Counter({7: 1}), Counter()]
        ]


if __name__ == "__main__":
    test_cm = TestDetectionConfusionMatrix()
    test_cm.test_run()
