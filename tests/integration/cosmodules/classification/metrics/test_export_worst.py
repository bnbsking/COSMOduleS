import glob

from cosmodules.classification.metrics.export_worst import export_worst_cases


def test_export_worst_cases_single_label():
    data_path_list = glob.glob("/app/tests/integration/input/classification/data/*.jpg")
    sorted_accuracy_index_list = [[0.0, 1], [1.0, 0], [1.0, 2]]
    save_folder = "/app/tests/integration/output/classification/metrics/export_worst_single_label"
    top_n = 2

    export_worst_cases(data_path_list, sorted_accuracy_index_list, save_folder, top_n)


def test_export_worst_cases_single_label_background():
    data_path_list = glob.glob("/app/tests/integration/input/classification/data/*.jpg")
    sorted_accuracy_index_list = [[1.0, 0], [1.0, 1], [1.0, 2]]
    save_folder = "/app/tests/integration/output/classification/metrics/export_worst_single_label_background"
    top_n = 2

    export_worst_cases(data_path_list, sorted_accuracy_index_list, save_folder, top_n)


def test_export_worst_cases_multi_label():
    data_path_list = glob.glob("/app/tests/integration/input/classification/data/*.jpg")
    sorted_accuracy_index_list = [[1.0, 0], [1.0, 1], [1.0, 2]]
    save_folder = "/app/tests/integration/output/classification/metrics/export_worst_multi_label"
    top_n = 2

    export_worst_cases(data_path_list, sorted_accuracy_index_list, save_folder, top_n)


if __name__ == "__main__":
    test_export_worst_cases_single_label()
    test_export_worst_cases_single_label_background()
    test_export_worst_cases_multi_label()
    