from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from adjustText import adjust_text

from .pareto import pareto_mask


_FAMILY_COLORS = {
    "CNN/pathway": "#2B6F8A",
    "Video Transformer": "#C2643B",
    "Foundation/pretrained": "#6B5B95",
}


def make_onset_resolved_figure(summary: pd.DataFrame, primary: pd.DataFrame, path: Path) -> None:
    """Plot Figure 3C from fixed-full-clip onset-resolved RSA results."""
    required = {"model", "time_ms", "onset_resolved_mean_r"}
    if not required <= set(summary) or not {"model", "whole_clip_mean_r"} <= set(primary):
        raise ValueError("Onset and primary RSA tables do not have the required columns")
    highlighted = primary.nlargest(5, "whole_clip_mean_r")["model"].tolist()
    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    colors = plt.get_cmap("tab10").colors
    for color, model in zip(colors, highlighted):
        values = summary[summary["model"] == model].sort_values("time_ms")
        if values.empty:
            raise ValueError(f"Onset-resolved RSA table is missing highlighted model {model}")
        axis.plot(values["time_ms"], values["onset_resolved_mean_r"], label=model, color=color, linewidth=1.5)
    other = summary[~summary["model"].isin(highlighted)]
    if not other.empty:
        median = other.groupby("time_ms", as_index=False)["onset_resolved_mean_r"].median()
        axis.plot(median["time_ms"], median["onset_resolved_mean_r"], "--", color="#777777", linewidth=1.5, label="Other endpoints median")
    axis.axvline(0, color="#333333", linestyle=":", linewidth=1)
    axis.axvspan(150, 250, color="#8096AB", alpha=.16, linewidth=0)
    axis.set(xlabel="Time from video onset (ms)", ylabel="Spearman RSA (r)", title="C. Onset-resolved correspondence to fixed full-clip model RDMs")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

def make_cross_axis_figure(data: pd.DataFrame, path: Path) -> None:
    data = data.copy()
    if "model" not in data:
        data["model"] = data["endpoint_id"]
    if "family" not in data:
        data["family"] = "Unspecified"
    family_colors = {**_FAMILY_COLORS, "Unspecified": "#666666"}
    pairs = (
        ("k400_top1", "rsa_r100", "A", "Accuracy and EEG correspondence"),
        ("k400_top1", "latency_ms", "B", "Accuracy and latency"),
        ("rsa_r100", "latency_ms", "C", "EEG correspondence and latency"),
        ("latency_ms", "memory_mb", "D", "Latency and memory"),
    )
    labels = {
        "k400_top1": "K400 top-1 accuracy (%)",
        "rsa_r100": "100-ms interval-integrated RSA (r)",
        "latency_ms": "Inference latency (ms)",
        "memory_mb": "Peak CUDA memory (MB)",
    }
    annotations = {
        "A": ("S3D", "SlowFast R50", "MViT-v1 B", "VideoMAE Base K400 fine-tuned"),
        "B": ("R3D-18", "MViT-v2 S", "Video Swin-B", "TimeSformer HR"),
        "C": ("S3D", "MViT-v1 B", "VideoMAE Large K400 fine-tuned", "TimeSformer HR"),
        "D": ("MC3-18", "S3D", "Video Swin-B", "VideoMAE Large K400 fine-tuned"),
    }
    label_names = {
        "VideoMAE Base K400 fine-tuned": "VideoMAE Base",
        "VideoMAE Large K400 fine-tuned": "VideoMAE Large",
    }
    # Match the manuscript's two-column figures: clear panels, sparse direct labels, and no decoding key.
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.0))
    for axis, (x, y, letter, title) in zip(axes.flat, pairs):
        for family, group in data.groupby("family", sort=False):
            axis.scatter(
                group[x], group[y], s=30, color=family_colors.get(family, "#666666"),
                edgecolor="white", linewidth=.6, alpha=.95, zorder=2,
            )

        if "latency_ms" in (x, y):
            getattr(axis, f"set_{'x' if x == 'latency_ms' else 'y'}scale")("log")
        if "memory_mb" in (x, y):
            getattr(axis, f"set_{'x' if x == 'memory_mb' else 'y'}scale")("log")
        axis.set(
            xlabel=labels[x], ylabel=labels[y],
        )
        axis.set_title(f"{letter}. {title}", loc="left", fontsize=8.8, fontweight="bold", color="#193755")
        axis.xaxis.label.set_fontsize(8)
        axis.yaxis.label.set_fontsize(8)
        axis.tick_params(labelsize=7)
        axis.grid(True, color="#D8DEE5", linewidth=.55, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        named_rows = data[data["model"].isin(annotations[letter])]
        texts = [
            axis.text(row[x], row[y], label_names.get(row["model"], row["model"]), fontsize=6.8, color="#263442", zorder=3)
            for _, row in named_rows.iterrows()
        ]
        adjust_text(
            texts, target_x=named_rows[x].to_numpy(), target_y=named_rows[y].to_numpy(),
            ax=axis, ensure_inside_axes=True, prevent_crossings=True,
            expand=(1.08, 1.15), force_text=(.35, .45), force_static=(.2, .3),
            force_pull=(.06, .08), pull_threshold=6, max_move=(12, 12), min_arrow_len=5,
            arrowprops={"arrowstyle": "-", "color": "#778493", "linewidth": .35, "alpha": .8},
        )

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
               markeredgecolor="white", markersize=6, label=family)
        for family, color in family_colors.items() if family in set(data["family"])
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
        ncol=len(handles), frameon=False, fontsize=8,
    )
    fig.subplots_adjust(left=.09, right=.99, bottom=.09, top=.88, hspace=.38, wspace=.23)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(data: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    data["pareto_latency"] = pareto_mask(data, ["k400_top1"], ["latency_ms"])
    data["pareto_memory"] = pareto_mask(data, ["k400_top1"], ["memory_mb"])
    data.to_csv(output / "pareto_endpoints.csv", index=False)
    make_cross_axis_figure(data, output / "cross_axis_relationships.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    scatter = None
    for axis, resource, flag, label in (
        (axes[0], "latency_ms", "pareto_latency", "Inference latency (ms)"),
        (axes[1], "memory_mb", "pareto_memory", "Peak allocated CUDA memory (MB)"),
    ):
        scatter = axis.scatter(
            data[resource], data["k400_top1"], c=data["rsa_r100"], cmap="viridis",
            s=data[flag].map({True: 75, False: 35}), edgecolors=data[flag].map({True: "black", False: "none"}),
        )
        frontier = data[data[flag]].sort_values(resource)
        axis.plot(frontier[resource], frontier["k400_top1"], color="black", linewidth=1, alpha=.65)
        axis.set_xscale("log")
        axis.set(xlabel=f"{label} (log scale)", title=f"Accuracy-{resource.removesuffix('_ms').removesuffix('_mb')}")
    axes[0].set_ylabel("K400 top-1 accuracy")
    fig.colorbar(scatter, ax=axes, label="100-ms interval-integrated RSA (r)", fraction=.025)
    fig.savefig(output / "pareto_landscape.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5))
    ranked = data.sort_values("rsa_r100", ascending=False)
    axis.bar(ranked["endpoint_id"], ranked["rsa_r100"], color="#315b7d")
    axis.set(ylabel="Interval-integrated RSA (r)", title="100-ms interval-integrated RSA")
    axis.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(output / "rsa_summary.png", dpi=180)
    plt.close(fig)
