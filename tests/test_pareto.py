import pandas as pd

from multi_axis_repro.pareto import pareto_mask


def test_pareto_frontier_handles_maximize_and_minimize():
    frame = pd.DataFrame({"score": [1, 2, 2], "cost": [3, 2, 4]})
    assert pareto_mask(frame, ["score"], ["cost"]).tolist() == [False, True, False]
