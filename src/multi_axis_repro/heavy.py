"""Lazy model execution adapters.

This module may be imported without Torch. Backends and weights are touched only
after a concrete execution command has validated its model options.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from .registry import MODELS, ModelSpec, get_model, validate_model_options

PUBLIC_MODELS = MODELS  # Compatibility for callers of the original six-model API.

_TV_DETAILS = {
    "R3D_18": ("r3d_18", "R3D_18_Weights", "fc"),
    "MC3_18": ("mc3_18", "MC3_18_Weights", "fc"),
    "R2Plus1D_18": ("r2plus1d_18", "R2Plus1D_18_Weights", "fc"),
    "S3D": ("s3d", "S3D_Weights", "classifier"),
    "MViT_V1_B": ("mvit_v1_b", "MViT_V1_B_Weights", "head"),
    "MViT_V2_S": ("mvit_v2_s", "MViT_V2_S_Weights", "head"),
    "Swin3D_T": ("swin3d_t", "Swin3D_T_Weights", "head"),
    "Swin3D_S": ("swin3d_s", "Swin3D_S_Weights", "head"),
    "Swin3D_B": ("swin3d_b", "Swin3D_B_Weights", "head"),
}

_HF_CLASSES = {
    "TimeSformer_K400_HF": ("TimesformerModel", "cls"),
    "TimeSformer_HR_K400_HF": ("TimesformerModel", "cls"),
    "ViViT_B_K400_HF": ("VivitModel", "cls"),
    "XCLIP_Base_P32_HF": ("XCLIPModel", "mean"),
    "VideoPrism_Base_F16_HF": ("AutoModel", "mean"),
    "VJEPA2_ViT_L_FPC64_HF": ("AutoModel", "mean"),
    # VideoMAE has patch tokens only; its Hugging Face classification head mean-pools them.
    "VideoMAE_Base_K400_HF": ("VideoMAEModel", "mean"),
    "VideoMAE_Large_K400_HF": ("VideoMAEModel", "mean"),
}


def _require(module: str, extra: str = "heavy"):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f"Model backend requires {module!r}; install with `pip install -e .[{extra}]`") from exc


def _torch():
    return _require("torch")


def _resize_normalize(frames, device: str, count: int, size: int, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225), time_first: bool = False):
    torch = _torch()
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).float().div(255.0).unsqueeze(0).to(device)
    video = torch.nn.functional.interpolate(video, size=(count, size, size), mode="trilinear", align_corners=False)
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    video = (video - mean_tensor) / std_tensor
    return video.permute(0, 2, 1, 3, 4) if time_first else video


def _feature_wrapper(model, pool: str):
    torch = _torch()

    class FeatureWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, **inputs):
            output = self.base(**inputs, return_dict=True)
            return _pool_hidden_state(_hidden_state(output), pool)

    return FeatureWrapper(model)


def _pool_hidden_state(hidden, pool: str):
    if hidden.ndim <= 2:
        return hidden
    if pool == "mean":
        return hidden.mean(dim=tuple(range(1, hidden.ndim - 1)))
    if pool == "cls":
        return hidden[:, 0]
    raise ValueError(f"Unsupported feature pooling method: {pool}")


def _xclip_feature_wrapper(model):
    torch = _torch()

    class XCLIPFeatureWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, **inputs):
            return _hidden_state(self.base.get_video_features(**inputs))

    return XCLIPFeatureWrapper(model)


def _hidden_state(output):
    torch = _torch()
    candidates = [output]
    for name in ("last_hidden_state", "video_embeds", "pooler_output", "vision_model_output", "video_model_output"):
        value = output.get(name) if isinstance(output, dict) else getattr(output, name, None)
        if value is not None:
            candidates.append(value)
    for value in candidates:
        if torch.is_tensor(value):
            return value
        nested = value.get("last_hidden_state") if isinstance(value, dict) else getattr(value, "last_hidden_state", None)
        if torch.is_tensor(nested):
            return nested
        if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
            return value[0]
    raise RuntimeError(f"Could not find a tensor representation in model output {type(output).__name__}")


def _logits(output):
    torch = _torch()
    if torch.is_tensor(output):
        return output
    value = output.get("logits") if isinstance(output, dict) else getattr(output, "logits", None)
    if torch.is_tensor(value):
        return value
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Could not find logits in model output {type(output).__name__}")


def _call(model, inputs):
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)


def _load_torchvision(spec: ModelSpec, device: str, features: bool):
    torch = _torch()
    video_models = _require("torchvision.models.video")
    constructor_name, weights_name, head = _TV_DETAILS[spec.id]
    weights = getattr(video_models, weights_name).DEFAULT
    model = getattr(video_models, constructor_name)(weights=weights)
    if features:
        setattr(model, head, torch.nn.Identity())
    transform = weights.transforms()

    def preprocess(frames):
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        return transform(tensor).unsqueeze(0).to(device)

    return model.eval().to(device), preprocess, str(weights)


def _load_pytorchvideo(spec: ModelSpec, device: str, features: bool):
    torch = _torch()
    hub = _require("pytorchvideo.models.hub")
    model = getattr(hub, spec.checkpoint)(pretrained=True)
    if features:
        model.blocks[-1].proj = torch.nn.Identity()
    slowfast = spec.checkpoint.startswith("slowfast_")

    def preprocess(frames):
        video = _resize_normalize(frames, device, 32 if slowfast else spec.frames, 256 if slowfast else 224)
        if not slowfast:
            return video
        top = (video.shape[-2] - 224) // 2
        left = (video.shape[-1] - 224) // 2
        video = video[:, :, :, top : top + 224, left : left + 224]
        indices = torch.linspace(0, video.shape[2] - 1, steps=8, device=device).long()
        return [torch.index_select(video, 2, indices), video]

    return model.eval().to(device), preprocess, f"pytorchvideo.models.hub.{spec.checkpoint}(pretrained=True)"


def _hf_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    return {key: options[key] for key in ("cache_dir", "revision", "local_files_only", "token", "trust_remote_code") if key in options}


def _load_huggingface(spec: ModelSpec, device: str, features: bool, options: dict[str, Any]):
    transformers = _require("transformers")
    processor_class = getattr(transformers, "AutoVideoProcessor", None) or getattr(transformers, "AutoImageProcessor", None)
    if processor_class is None:
        raise RuntimeError("This model requires a Transformers version with AutoVideoProcessor or AutoImageProcessor")
    kwargs = _hf_kwargs(options)
    processor = processor_class.from_pretrained(spec.checkpoint, **kwargs)
    if features:
        class_name, pool = _HF_CLASSES[spec.id]
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            raise RuntimeError(f"Transformers does not provide {class_name}; upgrade the `heavy` dependencies")
        base_model = model_class.from_pretrained(spec.checkpoint, **kwargs)
        model = _xclip_feature_wrapper(base_model) if spec.id == "XCLIP_Base_P32_HF" else _feature_wrapper(base_model, pool)
    else:
        model = transformers.AutoModelForVideoClassification.from_pretrained(spec.checkpoint, **kwargs)

    def preprocess(frames):
        try:
            encoded = processor(videos=list(frames), return_tensors="pt")
        except TypeError:
            encoded = processor(list(frames), return_tensors="pt")
        return {key: value.to(device) for key, value in encoded.items()}

    return model.eval().to(device), preprocess, spec.checkpoint


def _checkpoint_state(torch, checkpoint: Path):
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only.
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint {checkpoint} is not a state-dict payload")
    for key in ("model_state", "model", "state_dict"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def _local_paths(options: dict[str, Any], repo_key: str) -> tuple[Path, Path]:
    repo = Path(options[repo_key]).expanduser().resolve()
    checkpoint = Path(options["checkpoint"]).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Model repository not found: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    return repo, checkpoint


def _load_local_videomae(spec: ModelSpec, device: str, options: dict[str, Any]):
    torch = _torch()
    repo, checkpoint = _local_paths(options, "videomae_repo")
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("modeling_finetune")
    finally:
        if sys.path[0] == str(repo):
            sys.path.pop(0)
    constructor = getattr(module, "vit_base_patch16_224")
    model = constructor(num_classes=400, all_frames=16, tubelet_size=2, use_mean_pooling=True)
    encoder_state = {}
    for key, value in _checkpoint_state(torch, checkpoint).items():
        if key.startswith("encoder."):
            clean = key.removeprefix("encoder.")
            if clean.startswith("norm."):
                clean = "fc_norm." + clean.removeprefix("norm.")
            encoder_state[clean] = value
    if not encoder_state:
        raise RuntimeError("VideoMAE pretraining checkpoint contains no encoder.* parameters")
    report = model.load_state_dict(encoder_state, strict=False)
    missing = [key for key in report.missing_keys if not key.startswith("head")]
    unexpected = [key for key in report.unexpected_keys if not key.startswith("head")]
    if missing or unexpected:
        raise RuntimeError(f"VideoMAE checkpoint mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}")
    model.head = torch.nn.Identity()

    def preprocess(frames):
        video = _resize_normalize(frames, device, 16, 256, mean=(0.5,) * 3, std=(0.5,) * 3)
        top = (video.shape[-2] - 224) // 2
        left = (video.shape[-1] - 224) // 2
        return video[:, :, :, top : top + 224, left : left + 224]

    return model.eval().to(device), preprocess, str(checkpoint)


def _load(model_name: str, device: str, features: bool, model_options: dict[str, Any] | None = None):
    spec = get_model(model_name)
    options = validate_model_options(model_name, model_options)
    if not features and not spec.k400_accuracy:
        raise ValueError(f"{model_name} is feature-only and has no canonical K400 classifier; use `features` or `efficiency`")
    if spec.backend == "torchvision":
        loaded = _load_torchvision(spec, device, features)
    elif spec.backend == "pytorchvideo":
        loaded = _load_pytorchvideo(spec, device, features)
    elif spec.backend == "huggingface":
        loaded = _load_huggingface(spec, device, features, options)
    elif spec.backend == "videomae-local":
        loaded = _load_local_videomae(spec, device, options)
    else:
        raise RuntimeError(f"Unsupported model backend: {spec.backend}")
    return (*loaded, spec.frames)


def _decode(path: Path, frames: int):
    import numpy as np

    cv2 = _require("cv2")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not decoded:
        raise RuntimeError(f"Could not decode frames from {path}")
    indices = np.linspace(0, len(decoded) - 1, frames).round().astype(int)
    return np.stack([decoded[index] for index in indices])


def _repeat_starting_frames(frames, proportion: float):
    """Repeat a clip's starting proportion until its original length is filled."""
    import math
    import numpy as np

    if not 0 < proportion <= 1:
        raise ValueError("proportion must be greater than 0 and at most 1")
    if len(frames) == 0:
        raise ValueError("frames must contain at least one frame")
    prefix_count = min(len(frames), math.ceil(len(frames) * proportion))
    return frames[np.arange(len(frames)) % prefix_count]


def _interval_integrated_features(features_by_time, bin_ms: float, time_origin_ms: float, stimulus_duration_ms: float):
    """Concatenate synchronized prefix features in EEG-bin order."""
    import numpy as np

    if bin_ms <= 0:
        raise ValueError("bin_ms must be greater than 0")
    end_ms = time_origin_ms + stimulus_duration_ms
    expected = [min(start + bin_ms, end_ms) for start in np.arange(time_origin_ms, end_ms, bin_ms)]
    available = sorted(features_by_time)
    selected = []
    for endpoint in expected:
        matches = [time for time in available if np.isclose(time, endpoint)]
        if len(matches) != 1:
            raise ValueError(f"Missing synchronized model features for the {endpoint:g}-ms bin endpoint")
        selected.append(features_by_time[matches[0]])
    return np.concatenate(selected, axis=1), expected


def _manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest.resolve().parent / path).resolve()


def extract(model_name: str, clips_csv: Path, output: Path, device: str, model_options: dict[str, Any] | None = None) -> None:
    validate_model_options(model_name, model_options)
    import numpy as np
    import pandas as pd

    torch = _torch()
    clips = pd.read_csv(clips_csv)
    if not {"video_id", "path"} <= set(clips):
        raise ValueError("Feature clip manifest must contain video_id,path")
    model, preprocess, weights, frames = _load(model_name, device, features=True, model_options=model_options)
    features = []
    with torch.inference_mode():
        for row in clips.itertuples(index=False):
            path = _manifest_path(clips_csv, row.path)
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}")
            features.append(_hidden_state(_call(model, preprocess(_decode(path, frames)))).detach().cpu().numpy().reshape(-1))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "features.npy", np.stack(features).astype(np.float32))
    np.save(output / "kept_video_ids.npy", clips["video_id"].to_numpy())
    spec = get_model(model_name)
    metadata = {"model": model_name, "family": spec.family, "source": spec.source, "weights": weights, "frames": frames, "spatial_input": spec.spatial_input, "device": device}
    (output / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_model_rdm(features_path: Path, ids_path: Path, output: Path) -> None:
    import numpy as np
    from .stats import rdm_vector

    features = np.load(features_path)
    ids = np.load(ids_path)
    if features.ndim != 2 or len(features) != len(ids):
        raise ValueError("features must be 2D with one row per kept_video_ids entry")
    mean, std = features.mean(0, keepdims=True), features.std(0, keepdims=True)
    std[std == 0] = 1
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "rdm_vector.npy", rdm_vector((features - mean) / std).astype(np.float32))
    np.save(output / "kept_video_ids.npy", ids)


def make_time_proportion_model_rdm(
    model_name: str,
    clips_csv: Path,
    output: Path,
    device: str,
    proportion: float,
    model_options: dict[str, Any] | None = None,
) -> None:
    """Build an RDM from clips whose starting frames repeat for the full duration.

    Each clip is first sampled to the endpoint's normal frame count. The requested
    starting proportion is then tiled back to that frame count before extraction.
    """
    if not 0 < proportion <= 1:
        raise ValueError("proportion must be greater than 0 and at most 1")
    validate_model_options(model_name, model_options)
    import math
    import numpy as np
    import pandas as pd
    from .stats import rdm_vector

    torch = _torch()
    clips = pd.read_csv(clips_csv)
    if not {"video_id", "path"} <= set(clips):
        raise ValueError("Feature clip manifest must contain video_id,path")
    if clips.empty:
        raise ValueError("Feature clip manifest must contain at least one clip")
    model, preprocess, weights, frame_count = _load(model_name, device, features=True, model_options=model_options)
    features = []
    with torch.inference_mode():
        for row in clips.itertuples(index=False):
            path = _manifest_path(clips_csv, row.path)
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}")
            frames = _repeat_starting_frames(_decode(path, frame_count), proportion)
            feature = _hidden_state(_call(model, preprocess(frames)))
            features.append(feature.detach().cpu().numpy().reshape(-1))
    features = np.stack(features)
    mean, std = features.mean(0, keepdims=True), features.std(0, keepdims=True)
    std[std == 0] = 1
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "rdm_vector.npy", rdm_vector((features - mean) / std).astype(np.float32))
    np.save(output / "kept_video_ids.npy", clips["video_id"].to_numpy())
    spec = get_model(model_name)
    metadata = {
        "model": model_name,
        "family": spec.family,
        "source": spec.source,
        "weights": weights,
        "frames": frame_count,
        "starting_proportion": proportion,
        "starting_frames": min(frame_count, math.ceil(frame_count * proportion)),
        "device": device,
    }
    (output / "time_proportion_rdm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_time_synced_model_rdms(
    model_name: str,
    clips_csv: Path,
    output: Path | None,
    device: str,
    timepoints_ms: list[float],
    stimulus_duration_ms: float,
    model_options: dict[str, Any] | None = None,
    time_origin_ms: float = 0,
    integrated_output: Path | None = None,
    integrated_bin_ms: list[float] | None = None,
) -> None:
    """Create interval-integrated RDMs and, optionally, prefix RDMs with one model load."""
    validate_model_options(model_name, model_options)
    if stimulus_duration_ms <= 0:
        raise ValueError("stimulus_duration_ms must be greater than 0")
    if output is None and integrated_output is None:
        raise ValueError("At least one of output or integrated_output must be provided")
    if (integrated_output is None) != (integrated_bin_ms is None):
        raise ValueError("integrated_output and integrated_bin_ms must be provided together")
    if integrated_bin_ms is not None and (not integrated_bin_ms or any(width <= 0 for width in integrated_bin_ms)):
        raise ValueError("integrated_bin_ms must contain positive bin widths")
    if not timepoints_ms or any(time <= time_origin_ms or time > time_origin_ms + stimulus_duration_ms for time in timepoints_ms):
        raise ValueError("timepoints_ms must be non-empty and lie after time_origin_ms and within stimulus_duration_ms")
    if len(timepoints_ms) != len(set(timepoints_ms)):
        raise ValueError("timepoints_ms must not contain duplicates")
    import math
    import numpy as np
    import pandas as pd
    from .stats import rdm_vector

    torch = _torch()
    clips = pd.read_csv(clips_csv)
    if not {"video_id", "path"} <= set(clips) or clips.empty:
        raise ValueError("Feature clip manifest must contain at least one video_id,path row")
    model, preprocess, weights, frame_count = _load(model_name, device, features=True, model_options=model_options)
    ordered_times = set(timepoints_ms)
    if integrated_bin_ms is not None:
        end_ms = time_origin_ms + stimulus_duration_ms
        for bin_ms in integrated_bin_ms:
            ordered_times.update(min(start + bin_ms, end_ms) for start in np.arange(time_origin_ms, end_ms, bin_ms))
    ordered_times = sorted(ordered_times)
    features_by_time = {time: [] for time in ordered_times}
    with torch.inference_mode():
        for row in clips.itertuples(index=False):
            path = _manifest_path(clips_csv, row.path)
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}")
            frames = _decode(path, frame_count)
            for time in ordered_times:
                proportion = (time - time_origin_ms) / stimulus_duration_ms
                repeated = _repeat_starting_frames(frames, proportion)
                features_by_time[time].append(_hidden_state(_call(model, preprocess(repeated))).detach().cpu().numpy().reshape(-1))
    spec = get_model(model_name)
    features_by_time = {time: np.stack(features) for time, features in features_by_time.items()}
    if output is not None:
        for time in ordered_times:
            features = features_by_time[time]
            mean, std = features.mean(0, keepdims=True), features.std(0, keepdims=True)
            std[std == 0] = 1
            rdm_dir = output / f"{time:g}ms" / model_name
            rdm_dir.mkdir(parents=True, exist_ok=True)
            np.save(rdm_dir / "rdm_vector.npy", rdm_vector((features - mean) / std).astype(np.float32))
            np.save(rdm_dir / "kept_video_ids.npy", clips["video_id"].to_numpy())
            metadata = {
                "model": model_name,
                "family": spec.family,
                "source": spec.source,
                "weights": weights,
                "frames": frame_count,
                "timepoint_ms": time,
                "time_origin_ms": time_origin_ms,
                "stimulus_duration_ms": stimulus_duration_ms,
                "starting_proportion": (time - time_origin_ms) / stimulus_duration_ms,
                "starting_frames": min(frame_count, math.ceil(frame_count * (time - time_origin_ms) / stimulus_duration_ms)),
                "prefix_padding": "tile available prefix frames to the full model input length",
                "device": device,
            }
            (rdm_dir / "time_synced_rdm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if integrated_output is not None and integrated_bin_ms is not None:
        for bin_ms in integrated_bin_ms:
            features, endpoints = _interval_integrated_features(features_by_time, bin_ms, time_origin_ms, stimulus_duration_ms)
            mean, std = features.mean(0, keepdims=True), features.std(0, keepdims=True)
            std[std == 0] = 1
            rdm_dir = integrated_output / f"{bin_ms:g}ms" / model_name
            rdm_dir.mkdir(parents=True, exist_ok=True)
            np.save(rdm_dir / "rdm_vector.npy", rdm_vector((features - mean) / std).astype(np.float32))
            np.save(rdm_dir / "kept_video_ids.npy", clips["video_id"].to_numpy())
            metadata = {
                "analysis": "temporally matched, interval-integrated RSA",
                "model": model_name,
                "family": spec.family,
                "source": spec.source,
                "weights": weights,
                "frames": frame_count,
                "bin_ms": bin_ms,
                "interval_endpoints_ms": endpoints,
                "intervals": len(endpoints),
                "feature_dimensions_per_interval": features.shape[1] // len(endpoints),
                "concatenated_feature_dimensions": features.shape[1],
                "time_origin_ms": time_origin_ms,
                "stimulus_duration_ms": stimulus_duration_ms,
                "prefix_padding": "tile available prefix frames to the full model input length",
                "device": device,
            }
            (rdm_dir / "interval_integrated_rdm_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def accuracy(model_name: str, clips_csv: Path, output: Path, device: str, model_options: dict[str, Any] | None = None) -> None:
    spec = get_model(model_name)
    if not spec.k400_accuracy:
        raise ValueError(f"{model_name} is feature-only and has no canonical K400 classifier; choose an endpoint marked accuracy=yes in `multi-axis-repro models`")
    validate_model_options(model_name, model_options)
    import pandas as pd

    torch = _torch()
    clips = pd.read_csv(clips_csv)
    if not {"path", "label_index"} <= set(clips):
        raise ValueError("Classification manifest must contain path,label_index in checkpoint output-index space")
    if clips.empty:
        raise ValueError("Classification manifest must contain at least one clip")
    model, preprocess, weights, frames = _load(model_name, device, features=False, model_options=model_options)
    top1 = top5 = 0
    with torch.inference_mode():
        for row in clips.itertuples(index=False):
            logits = _logits(_call(model, preprocess(_decode(_manifest_path(clips_csv, row.path), frames))))[0]
            indices = logits.topk(min(5, logits.numel())).indices.cpu().tolist()
            top1 += indices[0] == int(row.label_index)
            top5 += int(row.label_index) in indices
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": model_name, "evaluated": len(clips), "weights": weights, "top1": top1 / len(clips), "top5": top5 / len(clips)}]).to_csv(output, index=False)


def efficiency(model_name: str, output: Path, device: str, warmup: int, repeats: int, seed: int, model_options: dict[str, Any] | None = None) -> None:
    validate_model_options(model_name, model_options)
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    import numpy as np
    import pandas as pd

    torch = _torch()
    spec = get_model(model_name)
    model, preprocess, weights, frames = _load(model_name, device, features=True, model_options=model_options)
    array = np.random.default_rng(seed).integers(0, 256, (frames, 256, 256, 3), dtype=np.uint8)
    inputs = preprocess(array)
    with torch.inference_mode():
        for _ in range(warmup):
            _call(model, inputs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(repeats):
            if device.startswith("cuda"):
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record(); _call(model, inputs); end_event.record(); torch.cuda.synchronize()
                durations.append(float(start_event.elapsed_time(end_event)))
            else:
                start = time.perf_counter(); _call(model, inputs); durations.append((time.perf_counter() - start) * 1000)
    row = {"model": model_name, "family": spec.family, "weights": weights, "frames": frames, "device": device, "parameters": sum(p.numel() for p in model.parameters()), "mean_inference_ms": np.mean(durations), "std_inference_ms": np.std(durations, ddof=1) if repeats > 1 else 0, "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.startswith("cuda") else np.nan}
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
