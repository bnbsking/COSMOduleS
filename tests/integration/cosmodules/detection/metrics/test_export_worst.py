import glob

from cosmodules.detection.metrics.export_worst import export_worst_cases


def test_export_worst_cases():
    data_path_list = glob.glob("/app/tests/integration/input/detection/data/general/*.jpg")
    ant_path = "/app/tests/integration/input/detection/prediction/general.json"
    sorted_accuracy_index_list = [(0.5, 0), (0.5, 1), (0.5, 2)]
    save_folder = "/app/tests/integration/output/detection/metrics/export_worst"
    top_n = 2

    export_worst_cases(data_path_list, ant_path, sorted_accuracy_index_list, save_folder, top_n)


if __name__ == "__main__":
    test_export_worst_cases()
    