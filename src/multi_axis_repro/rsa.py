from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

from .stats import holm_adjust, normalized_ranks, rdm_vector, square_rdm


def _mean_ci(scores: np.ndarray) -> tuple[float, float, float]:
    z = np.arctanh(np.clip(scores, -0.999999, 0.999999))
    mean = z.mean()
    sem = z.std(ddof=1) / np.sqrt(len(z))
    critical = t.ppf(0.975, len(z) - 1)
    return tuple(float(np.tanh(value)) for value in (mean, mean - critical * sem, mean + critical * sem))


def whole_clip(
    eeg_path: Path,
    video_ids_path: Path,
    rdm_root: Path,
    output: Path,
    bin_ms: int,
    time_start_ms: float,
    time_end_ms: float,
    epoch_ms: tuple[float, float],
    participant_count: int | None = None,
) -> None:
    eeg = np.load(eeg_path, mmap_mode="r")
    if eeg.ndim != 4:
        raise ValueError("EEG must be subjects x stimuli x channels x time; transpose it before this command")
    if participant_count is not None:
        eeg = eeg[:participant_count]
    video_ids = np.load(video_ids_path)
    if eeg.shape[1] != len(video_ids):
        raise ValueError("EEG stimulus count and video_ids length differ")
    times = np.linspace(time_start_ms, time_end_ms, eeg.shape[-1])
    starts = np.arange(epoch_ms[0], epoch_ms[1], bin_ms)
    bins = [np.flatnonzero((times >= start) & (times < min(start + bin_ms, epoch_ms[1]))) for start in starts]
    if not bins or any(len(indices) == 0 for indices in bins):
        raise ValueError("Requested epoch/binning contains empty bins; check sample rate and epoch")
    eeg_vectors = []
    for subject in range(eeg.shape[0]):
        pattern = np.stack([np.take(eeg[subject], indices, axis=-1).mean(axis=-1) for indices in bins], axis=-1).reshape(eeg.shape[1], -1)
        mean, std = pattern.mean(0, keepdims=True), pattern.std(0, keepdims=True)
        std[std == 0] = 1
        eeg_vectors.append(rdm_vector((pattern - mean) / std))
    eeg_vectors = np.stack(eeg_vectors)
    ranked_eeg = np.stack([normalized_ranks(vector) for vector in eeg_vectors])
    scores, rows = {}, []
    for directory in sorted(rdm_root.iterdir()):
        if not directory.is_dir() or not (directory / "rdm_vector.npy").exists():
            continue
        if not np.array_equal(np.load(directory / "kept_video_ids.npy"), video_ids):
            raise ValueError(f"Stimulus IDs do not match for {directory.name}")
        model_scores = ranked_eeg @ normalized_ranks(np.load(directory / "rdm_vector.npy"))
        scores[directory.name] = model_scores
        mean, low, high = _mean_ci(model_scores)
        try:
            statistic, p_value = wilcoxon(model_scores)
        except ValueError:
            statistic, p_value = np.nan, 1.0
        rows.append({"model": directory.name, "participants": len(eeg_vectors), "whole_clip_mean_r": mean, "ci95_low_r": low, "ci95_high_r": high, "wilcoxon_vs_zero_stat": statistic, "p_vs_zero_uncorrected": p_value})
    if not rows:
        raise ValueError(f"No model RDM directories found under {rdm_root}")
    for row, p_value in zip(rows, holm_adjust(row["p_vs_zero_uncorrected"] for row in rows)):
        row["p_vs_zero_holm"] = p_value
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("whole_clip_mean_r", ascending=False).to_csv(output / "whole_clip_rsa_summary.csv", index=False)
    np.save(output / "whole_clip_eeg_rdm_vectors.npy", eeg_vectors.astype(np.float32))
    np.save(output / "kept_video_ids.npy", video_ids)
    pair_rows = []
    for a, b in combinations(sorted(scores), 2):
        statistic, p_value = wilcoxon(scores[a], scores[b])
        pair_rows.append({"model_a": a, "model_b": b, "wilcoxon_stat": statistic, "p_uncorrected": p_value})
    if pair_rows:
        for row, p_value in zip(pair_rows, holm_adjust(row["p_uncorrected"] for row in pair_rows)):
            row["p_holm"] = p_value
        pd.DataFrame(pair_rows).to_csv(output / "pairwise_whole_clip_wilcoxon.csv", index=False)
    metadata = {
        "analysis": "temporally matched, interval-integrated RSA",
        "model_representation": "concatenated synchronized prefix features across EEG-bin endpoints",
        "participants": eeg.shape[0],
        "stimuli": eeg.shape[1],
        "channels": eeg.shape[2],
        "bin_ms": bin_ms,
        "time_start_ms": time_start_ms,
        "time_end_ms": time_end_ms,
        "timepoints": eeg.shape[-1],
        "epoch_ms": list(epoch_ms),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def onset_resolved(
    eeg_path: Path,
    video_ids_path: Path,
    rdm_root: Path,
    output: Path,
    time_start_ms: float,
    time_end_ms: float,
    epoch_ms: tuple[float, float],
    participant_count: int | None = None,
    neighborhood_samples: int = 5,
) -> None:
    """Compare centered EEG neighborhoods with each model's fixed full-clip RDM."""
    eeg = np.load(eeg_path, mmap_mode="r")
    if eeg.ndim != 4:
        raise ValueError("EEG must be subjects x stimuli x channels x time; transpose it before this command")
    if participant_count is not None:
        eeg = eeg[:participant_count]
    video_ids = np.load(video_ids_path)
    if eeg.shape[1] != len(video_ids):
        raise ValueError("EEG stimulus count and video_ids length differ")
    if neighborhood_samples < 1 or neighborhood_samples % 2 == 0:
        raise ValueError("neighborhood_samples must be a positive odd integer")
    times = np.linspace(time_start_ms, time_end_ms, eeg.shape[-1])
    radius = neighborhood_samples // 2
    centers = np.flatnonzero((times >= epoch_ms[0]) & (times <= epoch_ms[1]))
    centers = centers[(centers >= radius) & (centers < eeg.shape[-1] - radius)]
    if not len(centers):
        raise ValueError("Requested epoch contains no complete centered EEG neighborhoods")
    model_rdms = {}
    for directory in sorted(rdm_root.iterdir()):
        if not directory.is_dir() or not (directory / "rdm_vector.npy").exists():
            continue
        if not np.array_equal(np.load(directory / "kept_video_ids.npy"), video_ids):
            raise ValueError(f"Stimulus IDs do not match for {directory.name}")
        model_rdms[directory.name] = normalized_ranks(np.load(directory / "rdm_vector.npy"))
    if not model_rdms:
        raise ValueError(f"No fixed full-clip model RDM directories found under {rdm_root}")
    rows, eeg_vectors = [], []
    for center in centers:
        indices = np.arange(center - radius, center + radius + 1)
        subject_vectors = []
        for subject in range(eeg.shape[0]):
            pattern = np.take(eeg[subject], indices, axis=-1).reshape(eeg.shape[1], -1)
            mean, std = pattern.mean(0, keepdims=True), pattern.std(0, keepdims=True)
            std[std == 0] = 1
            subject_vectors.append(rdm_vector((pattern - mean) / std))
        subject_vectors = np.stack(subject_vectors)
        eeg_vectors.append(subject_vectors)
        ranked_eeg = np.stack([normalized_ranks(vector) for vector in subject_vectors])
        for model, model_rdm in model_rdms.items():
            scores = ranked_eeg @ model_rdm
            mean, low, high = _mean_ci(scores)
            rows.append({"model": model, "time_ms": times[center], "participants": len(subject_vectors), "onset_resolved_mean_r": mean, "ci95_low_r": low, "ci95_high_r": high})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["model", "time_ms"]).to_csv(output / "onset_resolved_rsa_summary.csv", index=False)
    np.save(output / "onset_resolved_eeg_rdm_vectors.npy", np.stack(eeg_vectors).astype(np.float32))
    np.save(output / "kept_video_ids.npy", video_ids)
    (output / "run_metadata.json").write_text(json.dumps({
        "participants": eeg.shape[0], "stimuli": eeg.shape[1], "channels": eeg.shape[2],
        "analysis": "onset-resolved EEG RSA against fixed full-clip model RDMs",
        "time_start_ms": time_start_ms, "time_end_ms": time_end_ms,
        "timepoints": eeg.shape[-1], "epoch_ms": list(epoch_ms),
        "neighborhood_samples": neighborhood_samples,
        "sample_interval_ms": float(times[1] - times[0]) if len(times) > 1 else None,
        "model_rdm_time_reference": "fixed full clip",
    }, indent=2), encoding="utf-8")


def _subset_rdm(square: np.ndarray, indices: np.ndarray, exclude_duplicates: bool = False) -> np.ndarray:
    rows, cols = np.triu_indices(len(indices), 1)
    source_rows, source_cols = indices[rows], indices[cols]
    if exclude_duplicates:
        keep = source_rows != source_cols
        source_rows, source_cols = source_rows[keep], source_cols[keep]
    return square[source_rows, source_cols]


def _stratified_sample(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for label in pd.unique(labels):
        members = np.flatnonzero(labels == label)
        sampled.extend(rng.choice(members, len(members), replace=True))
    return np.asarray(sampled, dtype=np.int64)


def generalization(
    rsa_root: Path,
    rdm_root: Path,
    stimulus_metadata: Path,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> None:
    eeg = np.load(rsa_root / "whole_clip_eeg_rdm_vectors.npy")
    ids = np.load(rsa_root / "kept_video_ids.npy")
    models, vectors = [], []
    for directory in sorted(rdm_root.iterdir()):
        if (directory / "rdm_vector.npy").exists():
            if not np.array_equal(np.load(directory / "kept_video_ids.npy"), ids):
                raise ValueError(f"Stimulus mismatch for {directory.name}")
            models.append(directory.name)
            vectors.append(np.load(directory / "rdm_vector.npy"))
    if not models:
        raise ValueError(f"No model RDM directories found under {rdm_root}")
    metadata = pd.read_csv(stimulus_metadata)
    if not {"video_id", "label"} <= set(metadata.columns):
        raise ValueError("Stimulus metadata must contain video_id,label")
    if metadata["video_id"].duplicated().any():
        raise ValueError("Stimulus metadata video_id values must be unique")
    metadata = metadata.set_index("video_id").reindex(ids)
    if metadata["label"].isna().any():
        raise ValueError("Stimulus metadata does not cover every RSA video ID")
    labels = metadata["label"].astype(str).to_numpy()
    eeg_ranked = np.stack([normalized_ranks(v) for v in eeg])
    model_ranked = np.stack([normalized_ranks(v) for v in vectors])
    participant_scores = model_ranked @ eeg_ranked.T
    observed = np.tanh(np.arctanh(np.clip(participant_scores, -.999999, .999999)).mean(1))
    rng = np.random.default_rng(seed)
    squares = np.stack([square_rdm(v) for v in vectors])
    exceed = np.zeros(len(models), int)
    exceed_max = np.zeros(len(models), int)
    for _ in range(permutations):
        permutation = rng.permutation(len(ids))
        permuted = np.stack([normalized_ranks(_subset_rdm(square, permutation)) for square in squares])
        null = np.tanh(np.arctanh(np.clip(permuted @ eeg_ranked.T, -.999999, .999999)).mean(1))
        exceed += np.abs(null) >= np.abs(observed)
        exceed_max += np.max(np.abs(null)) >= np.abs(observed)
    bootstrap = np.empty((bootstraps, len(models)))
    for index in range(bootstraps):
        stimuli = _stratified_sample(labels, rng)
        subjects = rng.integers(0, eeg.shape[0], eeg.shape[0])
        sampled_models = np.stack([normalized_ranks(_subset_rdm(square, stimuli, True)) for square in squares])
        sampled_eeg = np.stack([
            normalized_ranks(_subset_rdm(square_rdm(eeg[subject]), stimuli, True))
            for subject in subjects
        ])
        bootstrap[index] = np.tanh(
            np.arctanh(np.clip(sampled_models @ sampled_eeg.T, -.999999, .999999)).mean(1)
        )
    rows = [{
        "model": model,
        "observed_fisher_mean_r": observed[i],
        "stimulus_permutation_p_two_sided": (exceed[i] + 1) / (permutations + 1),
        "stimulus_permutation_p_maxT_fwer": (exceed_max[i] + 1) / (permutations + 1),
        "crossed_bootstrap_ci95_low_r": np.quantile(bootstrap[:, i], .025),
        "crossed_bootstrap_ci95_high_r": np.quantile(bootstrap[:, i], .975),
        "crossed_bootstrap_probability_r_above_zero": np.mean(bootstrap[:, i] > 0),
    } for i, model in enumerate(models)]
    pd.DataFrame(rows).to_csv(rsa_root / "whole_clip_generalization_inference.csv", index=False)
    np.save(rsa_root / "whole_clip_crossed_bootstrap_distributions.npy", bootstrap.astype(np.float32))
    (rsa_root / "whole_clip_generalization_metadata.json").write_text(json.dumps({
        "permutations": permutations,
        "bootstraps": bootstraps,
        "seed": seed,
        "permutation": "stimulus-identity permutation with maximum absolute statistic across endpoints",
        "bootstrap": "participants and stimuli resampled; stimuli stratified within action label; duplicate-identity pairs excluded",
    }, indent=2), encoding="utf-8")
