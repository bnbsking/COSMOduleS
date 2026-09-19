import glob

from cosmodules.segmentation.metrics.export_worst import export_worst_cases


def test_export_worst_cases():
    data_path_list = glob.glob("/app/tests/integration/input/segmentation/data/general/*.jpg")
    ant_path = "/app/tests/integration/input/segmentation/prediction/instance/general.json"
    sorted_accuracy_index_list = [(0.333, 0)]
    save_folder = "/app/tests/integration/output/segmentation/metrics/export_worst"
    top_n = 1

    export_worst_cases(data_path_list, ant_path, sorted_accuracy_index_list, save_folder, top_n)


if __name__ == "__main__":
    test_export_worst_cases()
    