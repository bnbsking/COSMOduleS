from cosmodules.segmentation.active_learning import (
    InstanceSegmentationActiveLearningByHFlip,
    SemanticSegmentationActiveLearning
)


class TestInstanceSegmentationActiveLearningByHFlip:
    InstanceSegmentationActiveLearningByHFlip(
        pred_path_1 = "/app/tests/integration/input/segmentation/prediction/instance/general.json",
        pred_path_2 = "/app/tests/integration/input/segmentation/prediction/instance_horizontal_flip/general.json",
        save_path = "/app/tests/integration/output/segmentation/active_learning/instance_general.json"
    )


class TestSemanticSegmentationActiveLearning:
    SemanticSegmentationActiveLearning(
        pred_path = "/app/tests/integration/input/segmentation/prediction/semantic/general.json",
        save_path = "/app/tests/integration/output/segmentation/active_learning/semantic_general.json"
    )


if __name__ == "__main__":
    TestInstanceSegmentationActiveLearningByHFlip()
    TestSemanticSegmentationActiveLearning()
