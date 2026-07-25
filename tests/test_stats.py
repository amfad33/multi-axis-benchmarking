import numpy as np

from multi_axis_repro.stats import holm_adjust, rdm_vector, square_rdm, upper_triangle


def test_rdm_condensed_vector_order_and_roundtrip():
    features = np.array([[0.0], [1.0], [3.0]])
    vector = rdm_vector(features, metric="euclidean")
    np.testing.assert_allclose(vector, [1.0, 3.0, 2.0])
    np.testing.assert_allclose(upper_triangle(square_rdm(vector)), vector)


def test_holm_adjustment_preserves_input_order_and_monotonicity():
    assert holm_adjust([0.03, 0.01, 0.04]) == [0.06, 0.03, 0.06]
