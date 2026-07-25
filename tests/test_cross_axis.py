import pandas as pd
import pytest

from multi_axis_repro.cross_axis import analyze, assemble


def fixtures():
    manifest = pd.DataFrame({"endpoint_id": ["a", "b"], "family": ["f1", "f2"], "provenance": ["public", "public"], "task_model": ["ta", "tb"], "efficiency_model": ["ea", "eb"], "rsa_model": ["ra", "rb"]})
    task = pd.DataFrame({"model": ["ta", "tb"], "k400_top1": [.5, .6]})
    efficiency = pd.DataFrame({"model": ["ea", "eb"], "mean_inference_ms": [1, 2], "peak_cuda_memory_mb": [3, 4]})
    rsa = pd.DataFrame({"model": ["ra", "rb"], "whole_clip_mean_r": [.01, .02]})
    return manifest, task, efficiency, rsa


def test_strict_join_schema_and_unmatched_detection():
    manifest, task, efficiency, rsa = fixtures()
    assert assemble(manifest, task, efficiency, rsa)["endpoint_id"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="unmatched"):
        assemble(manifest, task.iloc[:1], efficiency, rsa)
    with pytest.raises(ValueError, match="duplicate"):
        assemble(manifest, pd.concat([task, task.iloc[:1]]), efficiency, rsa)


def test_low_replicate_inference_is_deterministic():
    frame = pd.DataFrame({
        "endpoint_id": list("abcdefgh"),
        "family": ["cnn"] * 4 + ["vit"] * 4,
        "provenance": ["public"] * 8,
        "k400_top1": [.4, .55, .5, .7, .65, .8, .75, .9],
        "rsa_r100": [.02, .01, .04, .03, .08, .05, .07, .06],
        "log_latency_ms": [0, .2, .4, .6, .8, 1, 1.2, 1.4],
        "log_memory_mb": [.1, .3, .2, .7, .6, 1.1, .9, 1.3],
    })
    first = analyze(frame, bootstrap=7, permutations=9, seed=42)
    second = analyze(frame, bootstrap=7, permutations=9, seed=42)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)
