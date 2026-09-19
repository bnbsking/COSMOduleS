from collections import Counter

import numpy as np

from cosmodules.segmentation.metrics.confusion_matrix import (
    SegmentationConfusionMatrix
)


class TestSegmentationConfusionMatrix:
    def test_run(self):
        num_classes = 3
        img_idx = 7
        detections = np.array([
            [50, 100, 100, 150, 0.85, 1],
            [150, 50, 190, 100, 0.8, 1]
        ])  # xmin, ymin, xmax, ymax, confidence, class_id
        labels = np.array([
            [1, 50, 100, 100, 150],
            [2, 150, 50, 190, 100],
            [1, 150, 175, 200, 200]
        ])  # class_id, x_min, y_min, x_max, y_max
        label_mask = np.load(
            "/app/tests/integration/input/segmentation/data/general/gt_filled_img1.npy",
            allow_pickle=True
        )
        prediction_mask = np.load(
            "/app/tests/integration/input/segmentation/prediction/instance/pd_filled_img1.npy",
            allow_pickle=True
        )
        
        cm = SegmentationConfusionMatrix(num_classes, img_idx=img_idx)
        cm.process_batch(
            detections,
            labels,
            prediction_mask,
            label_mask
        )

        confusion = cm.get_confusion().tolist()
        confusion_with_img_indices = cm.get_confusion_with_img_indices()
        assert confusion == [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        assert confusion_with_img_indices == [
            [Counter(), Counter(), Counter()],
            [Counter({7: 1}), Counter({7: 1}), Counter()],
            [Counter({7: 1}), Counter(), Counter()]
        ]


if __name__ == "__main__":
    obj = TestSegmentationConfusionMatrix()
    obj.test_run()
