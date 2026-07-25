from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_mask(frame: pd.DataFrame, maximize: list[str], minimize: list[str]) -> np.ndarray:
    columns = maximize + minimize
    if not columns or frame[columns].isna().any().any():
        raise ValueError("Pareto columns must exist and contain no missing values")
    values = frame[columns].to_numpy(float, copy=True)
    values[:, len(maximize):] *= -1
    keep = np.ones(len(frame), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        dominates[index] = False
        keep[index] = not dominates.any()
    return keep
