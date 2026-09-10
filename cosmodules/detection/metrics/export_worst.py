import os
import shutil
from typing import List, Tuple

from cosmodules.detection.visualization.show import show_general


def export_worst_cases(
        data_path_list: List[str],
        ant_path: str,
        sorted_accuracy_index_list: List[Tuple[float, int]],
        save_folder: str,
        top_n: int | None = None
    ):
    os.makedirs(os.path.join(save_folder, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "visualized"), exist_ok=True)
    acc_indices = sorted_accuracy_index_list[:top_n]
    for acc, idx in acc_indices:
        filename = os.path.basename(data_path_list[idx])
        save_path = os.path.join(save_folder, "raw", f"{acc}_{filename}")
        shutil.copy(data_path_list[idx], save_path)

        vis_save_path = os.path.join(save_folder, "visualized", f"{acc}_{filename}")
        show_general(data_path_list[idx], ant_path, vis_save_path)
