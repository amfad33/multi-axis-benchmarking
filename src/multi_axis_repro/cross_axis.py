from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .stats import holm_adjust

REQUIRED_MANIFEST = {"endpoint_id", "family", "provenance", "task_model", "efficiency_model", "rsa_model"}
PAIRS = (
    ("k400_top1", "rsa_r100", "Accuracy-RSA"),
    ("k400_top1", "log_latency_ms", "Accuracy-latency"),
    ("k400_top1", "log_memory_mb", "Accuracy-memory"),
    ("rsa_r100", "log_latency_ms", "RSA-latency"),
    ("rsa_r100", "log_memory_mb", "RSA-memory"),
    ("log_latency_ms", "log_memory_mb", "Latency-memory"),
)


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _unique(frame: pd.DataFrame, column: str, label: str) -> None:
    duplicates = frame.loc[frame[column].duplicated(False), column].astype(str).unique()
    if len(duplicates):
        raise ValueError(f"{label} has duplicate {column} values: {duplicates.tolist()}")


def assemble(
    manifest: pd.DataFrame,
    task: pd.DataFrame,
    efficiency: pd.DataFrame,
    rsa: pd.DataFrame,
) -> pd.DataFrame:
    """Strictly assemble endpoint axes from versioned tables, never documents."""
    _require(manifest, REQUIRED_MANIFEST, "alias manifest")
    _require(task, {"model", "k400_top1"}, "task table")
    _require(efficiency, {"model", "mean_inference_ms", "peak_cuda_memory_mb"}, "efficiency table")
    _require(rsa, {"model", "whole_clip_mean_r"}, "RSA table")
    for frame, column, label in (
        (manifest, "endpoint_id", "manifest"),
        (manifest, "task_model", "manifest task aliases"),
        (manifest, "efficiency_model", "manifest efficiency aliases"),
        (manifest, "rsa_model", "manifest RSA aliases"),
        (task, "model", "task table"),
        (efficiency, "model", "efficiency table"),
        (rsa, "model", "RSA table"),
    ):
        _unique(frame, column, label)

    result = manifest.copy()
    result = result.merge(
        task[["model", "k400_top1"]].rename(columns={"model": "task_model"}),
        on="task_model", how="left", validate="one_to_one",
    )
    result = result.merge(
        efficiency[["model", "mean_inference_ms", "peak_cuda_memory_mb"]].rename(
            columns={"model": "efficiency_model", "mean_inference_ms": "latency_ms", "peak_cuda_memory_mb": "memory_mb"}
        ), on="efficiency_model", how="left", validate="one_to_one",
    )
    result = result.merge(
        rsa[["model", "whole_clip_mean_r"]].rename(columns={"model": "rsa_model", "whole_clip_mean_r": "rsa_r100"}),
        on="rsa_model", how="left", validate="one_to_one",
    )
    value_columns = ["k400_top1", "latency_ms", "memory_mb", "rsa_r100"]
    missing = result.loc[result[value_columns].isna().any(axis=1), ["endpoint_id"] + value_columns]
    if not missing.empty:
        raise ValueError("Strict join left unmatched endpoints:\n" + missing.to_string(index=False))
    if (result[["latency_ms", "memory_mb"]] <= 0).any().any():
        raise ValueError("Latency and memory must be positive")
    result["log_latency_ms"] = np.log10(result["latency_ms"])
    result["log_memory_mb"] = np.log10(result["memory_mb"])
    return result.sort_values("endpoint_id").reset_index(drop=True)


def analyze(data: pd.DataFrame, bootstrap: int, permutations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if bootstrap < 1 or permutations < 1:
        raise ValueError("bootstrap and permutations must be positive")
    results, loo_rows, family_rows, p_values = [], [], [], []
    for pair_index, (x, y, label) in enumerate(PAIRS):
        values = data[[x, y]].to_numpy(float)
        observed = float(spearmanr(values[:, 0], values[:, 1]).statistic)
        rng = np.random.default_rng(seed + pair_index)
        boot = []
        for _ in range(bootstrap):
            sample = values[rng.integers(0, len(values), len(values))]
            estimate = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
            if np.isfinite(estimate):
                boot.append(estimate)
        if not boot:
            raise ValueError(f"No finite bootstrap estimates for {label}")
        low, high = np.quantile(boot, [0.025, 0.975])
        rng = np.random.default_rng(seed + 100 + pair_index)
        exceed = sum(
            abs(float(spearmanr(values[:, 0], rng.permutation(values[:, 1])).statistic)) >= abs(observed)
            for _ in range(permutations)
        )
        p_raw = (exceed + 1) / (permutations + 1)
        tau, tau_p = kendalltau(values[:, 0], values[:, 1])
        public = data[data["provenance"] == "established public"]
        public_rho = float(spearmanr(public[x], public[y]).statistic) if len(public) >= 3 else np.nan
        loo = []
        for endpoint in data["endpoint_id"]:
            subset = data[data["endpoint_id"] != endpoint]
            estimate = float(spearmanr(subset[x], subset[y]).statistic)
            loo.append((endpoint, estimate))
            loo_rows.append({"association": label, "omitted_endpoint": endpoint, "n": len(subset), "spearman_rho": estimate})
        for omitted in sorted(data["family"].unique()):
            subset = data[data["family"] != omitted]
            family_rows.append({"association": label, "omitted_family": omitted, "n": len(subset), "spearman_rho": float(spearmanr(subset[x], subset[y]).statistic)})
        influential = max(loo, key=lambda item: abs(item[1] - observed))
        results.append({
            "association": label, "x": x, "y": y, "n": len(data), "spearman_rho": observed,
            "bootstrap_ci95_low": low, "bootstrap_ci95_high": high, "permutation_p_raw": p_raw,
            "kendall_tau": float(tau), "kendall_p": float(tau_p),
            "public_only_n": len(public), "public_only_rho": public_rho,
            "loo_min_rho": min(v for _, v in loo),
            "loo_max_rho": max(v for _, v in loo), "most_influential_endpoint": influential[0],
        })
        p_values.append(p_raw)
    for row, adjusted in zip(results, holm_adjust(p_values)):
        row["permutation_p_holm"] = adjusted
    return pd.DataFrame(results), pd.DataFrame(loo_rows), pd.DataFrame(family_rows)


def write_analysis(data: pd.DataFrame, output: Path, bootstrap: int, permutations: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    results, loo, families = analyze(data, bootstrap, permutations, seed)
    data.to_csv(output / "cross_axis_endpoint_manifest.csv", index=False)
    results.to_csv(output / "cross_axis_robust_associations.csv", index=False)
    loo.to_csv(output / "cross_axis_leave_one_endpoint_out.csv", index=False)
    families.to_csv(output / "cross_axis_leave_one_family_out.csv", index=False)
    metadata = {"common_subset_n": len(data), "bootstrap_replicates": bootstrap, "permutations": permutations, "seed": seed, "input_kind": "externally supplied CSV endpoints"}
    (output / "cross_axis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
