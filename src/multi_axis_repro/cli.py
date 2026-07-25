from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .registry import MODEL_IDS, MODELS, get_model, validate_model_options


_CONFIG_PATHS = (
    "eeg_matrix",
    "video_ids",
    "feature_manifest",
    "stimulus_metadata",
    "model_rdm_root",
    "classification_manifest",
    "alias_manifest",
    "output_root",
)


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _options_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model options JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--model-options must contain one JSON object for the selected model")
    return value


def _validated_options(model: str, path: Path | None) -> dict[str, Any]:
    return validate_model_options(model, _options_file(path))


def _print_models(as_json: bool, table_format: str) -> None:
    rows = [spec.as_dict() for spec in MODELS.values()]
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    headers = ("ID", "Backend", "Frames", "Input", "Features", "Accuracy", "Efficiency", "Required options")
    values = [
        (spec.id, spec.backend, str(spec.frames), spec.spatial_input, "yes" if spec.features else "no", "yes" if spec.k400_accuracy else "no", "yes" if spec.efficiency else "no", ", ".join(spec.required_options) or "-")
        for spec in MODELS.values()
    ]
    if table_format == "markdown":
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join("---" for _ in headers) + " |")
        for row in values:
            print("| " + " | ".join(row) + " |")
        return
    widths = [max(len(headers[index]), *(len(row[index]) for row in values)) for index in range(len(headers))]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in values:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def cross_axis_command(args: argparse.Namespace):
    import pandas as pd
    from .cross_axis import assemble, write_analysis

    data = assemble(pd.read_csv(args.manifest), pd.read_csv(args.task), pd.read_csv(args.efficiency), pd.read_csv(args.rsa))
    write_analysis(data, args.output, args.bootstrap, args.permutations, args.seed)
    return data


def _full_config(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config_path = config_path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Full-run config must contain one JSON object")
    required = ("eeg_matrix", "video_ids", "feature_manifest", "stimulus_metadata", "model_rdm_root", "classification_manifest", "alias_manifest", "output_root", "models")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Full-run config missing values: {missing}; start from configs/raw_rerun.example.json")
    models = config["models"]
    if not isinstance(models, list) or not models or not all(isinstance(model, str) for model in models):
        raise ValueError("Full-run `models` must be a non-empty JSON array of endpoint IDs")
    if len(models) != len(set(models)):
        raise ValueError("Full-run `models` contains duplicate endpoint IDs")
    for model in models:
        get_model(model)
    raw_options = config.get("model_options", {})
    if not isinstance(raw_options, dict):
        raise ValueError("Full-run `model_options` must be an object keyed by endpoint ID")
    extra = sorted(set(raw_options) - set(models))
    if extra:
        raise ValueError(f"model_options contains endpoints not selected in models: {extra}")
    base_dir = Path(config.get("base_dir", ".")).expanduser()
    if not base_dir.is_absolute():
        base_dir = config_path.parent / base_dir
    base_dir = base_dir.resolve()
    config = dict(config)
    for key in _CONFIG_PATHS:
        config[key] = _resolve_path(config[key], base_dir)
    options = {}
    for model in models:
        options[model] = validate_model_options(model, raw_options.get(model))
        for key in ("videomae_repo", "checkpoint", "cache_dir"):
            if key in options[model]:
                options[model][key] = _resolve_path(options[model][key], base_dir)
    return config, options


def full_run(config_path: Path) -> None:
    import numpy as np
    import pandas as pd
    from .cross_axis import assemble, write_analysis
    from .figures import make_figures, make_onset_resolved_figure
    from .heavy import accuracy, efficiency, extract, make_model_rdm, make_time_synced_model_rdms
    from .rsa import generalization, onset_resolved, whole_clip

    config, model_options = _full_config(config_path)
    output = Path(config["output_root"])
    clips = Path(config["feature_manifest"])
    if not clips.exists():
        raise FileNotFoundError(f"Create {clips} with columns video_id,path in EEG stimulus order")
    accuracy_models = [model for model in config["models"] if MODELS[model].k400_accuracy]
    epoch_ms = tuple(config.get("epoch_ms", [0, 3000]))
    onset_timepoints = sorted({
        min(start + bin_ms, epoch_ms[1])
        for bin_ms in (50, 100, 200)
        for start in np.arange(epoch_ms[0], epoch_ms[1], bin_ms)
    })
    for model in config["models"]:
        feature_dir = output / "features" / model
        rdm_dir = Path(config["model_rdm_root"]) / model
        options = model_options[model]
        extract(model, clips, feature_dir, config.get("device", "cuda"), options)
        make_model_rdm(feature_dir / "features.npy", feature_dir / "kept_video_ids.npy", rdm_dir)
        make_time_synced_model_rdms(
            model, clips, None,
            config.get("device", "cuda"), onset_timepoints, epoch_ms[1] - epoch_ms[0],
            options, epoch_ms[0], Path(config["model_rdm_root"]) / "interval_integrated", [50, 100, 200],
        )
        if MODELS[model].k400_accuracy:
            accuracy(model, Path(config["classification_manifest"]), output / "accuracy" / f"{model}.csv", config.get("device", "cuda"), options)
        efficiency(model, output / "efficiency" / f"{model}.csv", config.get("device", "cuda"), config.get("warmup", 3), config.get("repeats", 10), config.get("benchmark_seed", 123), options)
    for bin_ms in (50, 100, 200):
        whole_clip(Path(config["eeg_matrix"]), Path(config["video_ids"]), Path(config["model_rdm_root"]) / "interval_integrated" / f"{bin_ms}ms", output / "rsa" / f"whole_clip_{bin_ms}ms", bin_ms, config.get("time_start_ms", -200), config.get("time_end_ms", 3400), epoch_ms, config.get("participant_count"))
    onset_epoch_ms = tuple(config.get("onset_epoch_ms", [-200, 800]))
    onset_resolved(
        Path(config["eeg_matrix"]), Path(config["video_ids"]), Path(config["model_rdm_root"]),
        output / "rsa" / "onset_resolved", config.get("time_start_ms", -200),
        config.get("time_end_ms", 3400), onset_epoch_ms, config.get("participant_count"),
        config.get("onset_neighborhood_samples", 5),
    )
    generalization(output / "rsa" / "whole_clip_100ms", Path(config["model_rdm_root"]) / "interval_integrated" / "100ms", Path(config["stimulus_metadata"]), config.get("permutations", 10000), config.get("bootstraps", 2000), config.get("seed", 20260718))
    rsa_table = pd.read_csv(output / "rsa" / "whole_clip_100ms" / "whole_clip_rsa_summary.csv")
    make_onset_resolved_figure(
        pd.read_csv(output / "rsa" / "onset_resolved" / "onset_resolved_rsa_summary.csv"),
        rsa_table,
        output / "figures" / "figure_3c_onset_resolved.png",
    )
    if not accuracy_models:
        return
    task = pd.concat([pd.read_csv(output / "accuracy" / f"{model}.csv") for model in accuracy_models], ignore_index=True)
    task = task[["model", "top1"]].rename(columns={"top1": "k400_top1"})
    efficiency_table = pd.concat([pd.read_csv(output / "efficiency" / f"{model}.csv") for model in accuracy_models], ignore_index=True)
    manifest = pd.read_csv(Path(config["alias_manifest"]))
    manifest = manifest[manifest["endpoint_id"].isin(accuracy_models)]
    if set(manifest["endpoint_id"]) != set(accuracy_models):
        missing_aliases = sorted(set(accuracy_models) - set(manifest["endpoint_id"]))
        raise ValueError(f"Alias manifest does not cover accuracy-capable configured models: {missing_aliases}")
    endpoints = assemble(manifest, task, efficiency_table, rsa_table)
    write_analysis(endpoints, output / "cross_axis", config.get("cross_axis_bootstraps", 10000), config.get("cross_axis_permutations", 20000), config.get("seed", 20260720))
    make_figures(endpoints, output / "figures")


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(prog="multi-axis-repro", description="Multi-axis reproducibility CLI")
    sub = main.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("models", help="list all model endpoints without loading model backends")
    listing.add_argument("--json", action="store_true", help="emit complete registry metadata as JSON")
    listing.add_argument("--format", choices=("plain", "markdown"), default="plain")
    task = sub.add_parser("accuracy", help="benchmark public checkpoint K400 task accuracy")
    task.add_argument("--model", required=True, choices=MODEL_IDS); task.add_argument("--clips", type=Path, required=True); task.add_argument("--output", type=Path, required=True); task.add_argument("--device", default="cuda"); task.add_argument("--model-options", type=Path)
    gpu = sub.add_parser("efficiency", help="benchmark feature-forward latency and GPU memory")
    gpu.add_argument("--model", required=True, choices=MODEL_IDS); gpu.add_argument("--output", type=Path, required=True); gpu.add_argument("--device", default="cuda"); gpu.add_argument("--warmup", type=int, default=2); gpu.add_argument("--repeats", type=int, default=5); gpu.add_argument("--seed", type=int, default=123); gpu.add_argument("--model-options", type=Path)
    features = sub.add_parser("features", help="extract public-model clip features")
    features.add_argument("--model", required=True, choices=MODEL_IDS); features.add_argument("--clips", type=Path, required=True); features.add_argument("--output", type=Path, required=True); features.add_argument("--device", default="cuda"); features.add_argument("--model-options", type=Path)
    rdm = sub.add_parser("model-rdm", help="generate a model RDM from feature rows")
    rdm.add_argument("--features", type=Path, required=True); rdm.add_argument("--video-ids", type=Path, required=True); rdm.add_argument("--output", type=Path, required=True)
    temporal_rdm = sub.add_parser("time-synced-model-rdms", help="generate synchronized onset and interval-integrated model RDMs")
    temporal_rdm.add_argument("--model", required=True, choices=MODEL_IDS); temporal_rdm.add_argument("--clips", type=Path, required=True); temporal_rdm.add_argument("--output", type=Path); temporal_rdm.add_argument("--timepoints-ms", type=float, nargs="+", required=True); temporal_rdm.add_argument("--stimulus-duration-ms", type=float, required=True); temporal_rdm.add_argument("--time-origin-ms", type=float, default=0); temporal_rdm.add_argument("--integrated-output", type=Path); temporal_rdm.add_argument("--integrated-bin-ms", type=float, nargs="+"); temporal_rdm.add_argument("--device", default="cuda"); temporal_rdm.add_argument("--model-options", type=Path)
    rsa = sub.add_parser("whole-clip-rsa", help="run whole-clip EEG RSA at 50, 100, or 200 ms")
    rsa.add_argument("--eeg", type=Path, required=True); rsa.add_argument("--video-ids", type=Path, required=True); rsa.add_argument("--model-rdms", type=Path, required=True); rsa.add_argument("--output", type=Path, required=True); rsa.add_argument("--bin-ms", type=int, choices=[50, 100, 200], required=True); rsa.add_argument("--time-start-ms", type=float, default=-200); rsa.add_argument("--time-end-ms", type=float, default=3400); rsa.add_argument("--epoch-ms", type=float, nargs=2, default=(0, 3000)); rsa.add_argument("--participants", type=int)
    onset = sub.add_parser("onset-resolved-rsa", help="compare centered EEG neighborhoods with fixed full-clip model RDMs")
    onset.add_argument("--eeg", type=Path, required=True); onset.add_argument("--video-ids", type=Path, required=True); onset.add_argument("--model-rdms", type=Path, required=True); onset.add_argument("--output", type=Path, required=True); onset.add_argument("--time-start-ms", type=float, default=-200); onset.add_argument("--time-end-ms", type=float, default=3400); onset.add_argument("--epoch-ms", type=float, nargs=2, default=(-200, 800)); onset.add_argument("--neighborhood-samples", type=int, default=5); onset.add_argument("--participants", type=int)
    gen = sub.add_parser("generalization", help="run max-statistic permutation and crossed participant/stimulus bootstrap")
    gen.add_argument("--rsa-root", type=Path, required=True); gen.add_argument("--model-rdms", type=Path, required=True); gen.add_argument("--stimulus-metadata", type=Path, required=True, help="CSV with video_id,label"); gen.add_argument("--permutations", type=int, default=10000); gen.add_argument("--bootstraps", type=int, default=2000); gen.add_argument("--seed", type=int, default=20260718)
    cross = sub.add_parser("cross-axis", help="strictly assemble CSV endpoints and run robust inference")
    cross.add_argument("--manifest", type=Path, required=True); cross.add_argument("--task", type=Path, required=True); cross.add_argument("--efficiency", type=Path, required=True); cross.add_argument("--rsa", type=Path, required=True); cross.add_argument("--output", type=Path, required=True); cross.add_argument("--bootstrap", type=int, default=10000); cross.add_argument("--permutations", type=int, default=20000); cross.add_argument("--seed", type=int, default=20260720)
    figures = sub.add_parser("figures", help="create Pareto and RSA figures from an assembled endpoint CSV")
    figures.add_argument("--endpoints", type=Path, required=True); figures.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="run the end-to-end pipeline from external inputs")
    run.add_argument("--config", type=Path, required=True)
    return main


def main() -> None:
    args = parser().parse_args()
    if args.command == "models":
        _print_models(args.json, args.format)
        return
    if args.command in {"accuracy", "efficiency", "features"}:
        if args.command == "accuracy" and not MODELS[args.model].k400_accuracy:
            raise ValueError(f"{args.model} is feature-only and has no canonical K400 classifier; use `models` to select an accuracy=yes endpoint")
        options = _validated_options(args.model, args.model_options)
        from .heavy import accuracy, efficiency, extract
        if args.command == "accuracy": accuracy(args.model, args.clips, args.output, args.device, options)
        elif args.command == "efficiency": efficiency(args.model, args.output, args.device, args.warmup, args.repeats, args.seed, options)
        else: extract(args.model, args.clips, args.output, args.device, options)
        return
    if args.command == "model-rdm":
        from .heavy import make_model_rdm
        make_model_rdm(args.features, args.video_ids, args.output)
    elif args.command == "time-synced-model-rdms":
        from .heavy import make_time_synced_model_rdms
        make_time_synced_model_rdms(args.model, args.clips, args.output, args.device, args.timepoints_ms, args.stimulus_duration_ms, _validated_options(args.model, args.model_options), args.time_origin_ms, args.integrated_output, args.integrated_bin_ms)
    elif args.command == "whole-clip-rsa":
        from .rsa import whole_clip
        whole_clip(args.eeg, args.video_ids, args.model_rdms, args.output, args.bin_ms, args.time_start_ms, args.time_end_ms, tuple(args.epoch_ms), args.participants)
    elif args.command == "onset-resolved-rsa":
        from .rsa import onset_resolved
        onset_resolved(args.eeg, args.video_ids, args.model_rdms, args.output, args.time_start_ms, args.time_end_ms, tuple(args.epoch_ms), args.participants, args.neighborhood_samples)
    elif args.command == "generalization":
        from .rsa import generalization
        generalization(args.rsa_root, args.model_rdms, args.stimulus_metadata, args.permutations, args.bootstraps, args.seed)
    elif args.command == "cross-axis": cross_axis_command(args)
    elif args.command == "figures":
        import pandas as pd
        from .figures import make_figures
        make_figures(pd.read_csv(args.endpoints), args.output)
    else: full_run(args.config)


if __name__ == "__main__":
    main()
