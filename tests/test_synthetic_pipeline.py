import json

import matplotlib
import numpy as np
import pandas as pd

from multi_axis_repro.figures import make_figures, make_onset_resolved_figure
from multi_axis_repro.heavy import _HF_CLASSES, _interval_integrated_features, _manifest_path, _pool_hidden_state, _repeat_starting_frames, make_model_rdm
from multi_axis_repro.rsa import generalization, onset_resolved, whole_clip
from multi_axis_repro.stats import rdm_vector


matplotlib.use("Agg")


def test_starting_frame_proportion_repeats_to_full_clip():
    frames = np.arange(10)
    np.testing.assert_array_equal(
        _repeat_starting_frames(frames, 0.2),
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    )
    np.testing.assert_array_equal(_repeat_starting_frames(frames, 1.0), frames)


def test_huggingface_videomae_uses_patch_token_mean_pooling():
    assert _HF_CLASSES["VideoMAE_Base_K400_HF"] == ("VideoMAEModel", "mean")
    assert _HF_CLASSES["VideoMAE_Large_K400_HF"] == ("VideoMAEModel", "mean")

    class HiddenState:
        def __init__(self, values):
            self.values = values
            self.ndim = values.ndim

        def mean(self, dim):
            return self.values.mean(axis=dim)

        def __getitem__(self, index):
            return self.values[index]

    patch_tokens = np.arange(24).reshape(2, 3, 4)
    np.testing.assert_array_equal(_pool_hidden_state(HiddenState(patch_tokens), "mean"), patch_tokens.mean(axis=1))


def test_time_synced_features_are_concatenated_in_eeg_bin_order():
    features = {
        200: np.asarray([[20, 21], [22, 23]]),
        100: np.asarray([[10, 11], [12, 13]]),
        300: np.asarray([[30, 31], [32, 33]]),
    }
    integrated, endpoints = _interval_integrated_features(features, 100, 0, 300)
    assert endpoints == [100, 200, 300]
    np.testing.assert_array_equal(integrated, [[10, 11, 20, 21, 30, 31], [12, 13, 22, 23, 32, 33]])


def test_data_pipeline_runs_on_synthetic_inputs(tmp_path):
    rng = np.random.default_rng(20260720)
    ids = np.asarray([f"clip-{index}" for index in range(6)])
    ids_path = tmp_path / "video_ids.npy"
    np.save(ids_path, ids)

    rdm_root = tmp_path / "model_rdms"
    for model in ("model-a", "model-b", "model-c"):
        features = tmp_path / f"{model}-features.npy"
        np.save(features, rng.normal(size=(len(ids), 8)))
        make_model_rdm(features, ids_path, rdm_root / model)

    eeg_path = tmp_path / "eeg.npy"
    np.save(eeg_path, rng.normal(size=(4, len(ids), 3, 12)).astype(np.float32))
    rsa_root = tmp_path / "rsa"
    whole_clip(eeg_path, ids_path, rdm_root, rsa_root, 100, -100, 500, (0, 300))

    metadata = tmp_path / "stimuli.csv"
    pd.DataFrame({"video_id": ids, "label": ["one"] * 3 + ["two"] * 3}).to_csv(metadata, index=False)
    generalization(rsa_root, rdm_root, metadata, permutations=3, bootstraps=4, seed=42)

    summary = pd.read_csv(rsa_root / "whole_clip_rsa_summary.csv")
    assert set(summary["model"]) == {"model-a", "model-b", "model-c"}
    assert len(pd.read_csv(rsa_root / "whole_clip_generalization_inference.csv")) == 3
    assert json.loads((rsa_root / "whole_clip_generalization_metadata.json").read_text())["seed"] == 42

    onset_root = tmp_path / "onset_rsa"
    onset_resolved(eeg_path, ids_path, rdm_root, onset_root, -100, 500, (0, 300), neighborhood_samples=5)
    onset_summary = pd.read_csv(onset_root / "onset_resolved_rsa_summary.csv")
    assert len(onset_summary) == 18
    np.testing.assert_allclose(sorted(onset_summary["time_ms"].unique()), np.linspace(-100, 500, 12)[2:8])
    run_metadata = json.loads((onset_root / "run_metadata.json").read_text())
    assert run_metadata["model_rdm_time_reference"] == "fixed full clip"
    assert run_metadata["neighborhood_samples"] == 5
    first_pattern = np.load(eeg_path)[0, :, :, :5].reshape(len(ids), -1)
    first_pattern = (first_pattern - first_pattern.mean(0, keepdims=True)) / np.where(first_pattern.std(0, keepdims=True) == 0, 1, first_pattern.std(0, keepdims=True))
    np.testing.assert_allclose(np.load(onset_root / "onset_resolved_eeg_rdm_vectors.npy")[0, 0], rdm_vector(first_pattern), rtol=1e-6)
    make_onset_resolved_figure(onset_summary, summary, tmp_path / "figure_3c.png")
    assert (tmp_path / "figure_3c.png").is_file()


def test_figures_and_manifest_relative_paths(tmp_path):
    manifest = tmp_path / "manifests" / "clips.csv"
    manifest.parent.mkdir()
    assert _manifest_path(manifest, "../videos/a.mp4") == (tmp_path / "videos" / "a.mp4").resolve()

    endpoints = pd.DataFrame({
        "endpoint_id": ["a", "b", "c"],
        "k400_top1": [0.5, 0.7, 0.6],
        "latency_ms": [10.0, 30.0, 15.0],
        "memory_mb": [100.0, 300.0, 150.0],
        "rsa_r100": [0.02, 0.04, 0.03],
    })
    output = tmp_path / "figures"
    make_figures(endpoints, output)
    assert {path.name for path in output.iterdir()} == {
        "cross_axis_relationships.png",
        "pareto_endpoints.csv",
        "pareto_landscape.png",
        "rsa_summary.png",
    }
