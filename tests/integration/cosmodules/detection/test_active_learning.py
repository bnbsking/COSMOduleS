from cosmodules.detection.active_learning import DetectionActiveLearningByHFlip


class TestDetectionActiveLearningByHFlip:
    DetectionActiveLearningByHFlip(
        pred_path_1 = "/app/tests/integration/input/detection/prediction/general.json",
        pred_path_2 = "/app/tests/integration/input/detection/prediction/general_horizontal_flip.json",
        save_path = "/app/tests/integration/output/detection/active_learning/general.json"
    )


if __name__ == "__main__":
    TestDetectionActiveLearningByHFlip()
    