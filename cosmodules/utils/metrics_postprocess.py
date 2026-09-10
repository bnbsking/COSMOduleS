from typing import Dict, List

import numpy as np

from cosmodules.utils.metrics import PRCurve


def deserialize(data: Dict | List | np.ndarray | int | float) -> Dict | List | int | float:
    if isinstance(data, dict):
        return {k: deserialize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deserialize(v) for v in data]
    elif isinstance(data, np.ndarray):
        return [deserialize(v) for v in data.tolist()]
    elif isinstance(data, PRCurve):
        return {
            "precision": [round(float(v), 3) for v in data.precision],
            "recall": [round(float(v), 3) for v in data.recall]
        }
    else:
        if "numpy" in str(type(data)):
            data = float(data)
        return round(data, 3) if isinstance(data, float) else data
    