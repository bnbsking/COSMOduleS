import os
import shutil
from typing import List, Tuple


def export_worst_cases(
        data_path_list: List[str],
        sorted_accuracy_index_list: List[Tuple[float, int]],
        save_folder: str,
        top_n: int | None = None
    ):
    os.makedirs(os.path.join(save_folder, "raw"), exist_ok=True)
    acc_indices = sorted_accuracy_index_list[:top_n]
    for acc, idx in acc_indices:
        filename = os.path.basename(data_path_list[idx])
        save_path = os.path.join(save_folder, "raw", f"{acc}_{filename}")
        shutil.copy(data_path_list[idx], save_path)
