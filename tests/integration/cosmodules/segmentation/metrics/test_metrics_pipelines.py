import json

import numpy as np

from cosmodules.segmentation.metrics.metrics_pipelines import (
    SegmentationLabel,
    SegmentationPrediction,
    InstanceSegmentationMetricsPipeline,
    SemanticSegmentationMetricsPipeline
)


class TestInstanceSegmentationMetricsPipeline:
    def test_run(self):
        num_classes = 3
        labels = [
            SegmentationLabel(
                segmentation_path="/app/tests/integration/input/segmentation/data/general/gt_filled_img1.npy",
                detection=np.array(
                    [
                        [1, 50, 100, 100, 150],
                        [2, 150, 50, 190, 100],
                        [1, 150, 175, 200, 200]
                    ]
                )
            )
        ]
        predictions = [
            SegmentationPrediction(
                segmentation_path="/app/tests/integration/input/segmentation/prediction/instance/pd_filled_img1.npy",
                detection=np.array(
                    [
                        [50, 100, 100, 150, 0.85, 1],
                        [150, 50, 190, 100, 0.80, 1]
                    ]
                )
            )
        ]

        pipeline = InstanceSegmentationMetricsPipeline()
        out = pipeline.run(num_classes, labels, predictions)
        print(out)
        
        with open("tests/integration/output/segmentation/metrics/metrics_instance.json", "w") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)


class TestSemanticSegmentationMetricsPipeline:
    def test_run(self):
        num_classes = 3
        labels = [
            SegmentationLabel(
                segmentation_path="/app/tests/integration/input/segmentation/data/general/gt_filled_img1.npy",
                detection=np.array([])
            )
        ]
        predictions = [
            SegmentationPrediction(
                segmentation_path="/app/tests/integration/input/segmentation/prediction/semantic/pd_filled_img1.npy",
                detection=np.array([])
            )
        ]

        pipeline = SemanticSegmentationMetricsPipeline()
        out = pipeline.run(num_classes, labels, predictions)
        print(out)
        
        with open("tests/integration/output/segmentation/metrics/metrics_semantic.json", "w") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    obj = TestInstanceSegmentationMetricsPipeline()
    obj.test_run()

    obj = TestSemanticSegmentationMetricsPipeline()
    obj.test_run()


