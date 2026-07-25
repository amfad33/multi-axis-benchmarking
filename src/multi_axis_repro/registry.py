"""Dependency-free metadata for the canonical model ecosystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display: str
    family: str
    backend: str
    source: str
    checkpoint: str
    frames: int
    spatial_input: str
    features: bool
    k400_accuracy: bool
    efficiency: bool
    required_options: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    option_keys: tuple[str, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            feature_support=self.features,
            k400_accuracy_support=self.k400_accuracy,
            efficiency_support=self.efficiency,
        )
        return data

    @property
    def feature_support(self) -> bool:
        return self.features

    @property
    def k400_accuracy_support(self) -> bool:
        return self.k400_accuracy

    @property
    def efficiency_support(self) -> bool:
        return self.efficiency


_TV = (
    ("R3D_18", "R3D-18", "3D CNN", "r3d_18", "R3D_18_Weights", 16, "112x112"),
    ("MC3_18", "MC3-18", "mixed 2D/3D CNN", "mc3_18", "MC3_18_Weights", 16, "112x112"),
    ("R2Plus1D_18", "R(2+1)D-18", "factorized 3D CNN", "r2plus1d_18", "R2Plus1D_18_Weights", 16, "112x112"),
    ("S3D", "S3D", "separable 3D CNN", "s3d", "S3D_Weights", 64, "224x224"),
    ("MViT_V1_B", "MViT V1-B", "multiscale video transformer", "mvit_v1_b", "MViT_V1_B_Weights", 16, "224x224"),
    ("MViT_V2_S", "MViT V2-S", "multiscale video transformer", "mvit_v2_s", "MViT_V2_S_Weights", 16, "224x224"),
    ("Swin3D_T", "Swin3D-T", "video Swin transformer", "swin3d_t", "Swin3D_T_Weights", 32, "224x224"),
    ("Swin3D_S", "Swin3D-S", "video Swin transformer", "swin3d_s", "Swin3D_S_Weights", 32, "224x224"),
    ("Swin3D_B", "Swin3D-B", "video Swin transformer", "swin3d_b", "Swin3D_B_Weights", 32, "224x224"),
)

_PTV = (
    ("SlowFast_R50", "SlowFast R50", "dual-rate pathway CNN", "slowfast_r50", 32),
    ("SlowFast_R101", "SlowFast R101", "dual-rate pathway CNN", "slowfast_r101", 32),
    ("Slow_R50", "Slow R50", "slow pathway 3D CNN", "slow_r50", 8),
    ("I3D_R50", "I3D-R50", "inflated 3D CNN", "i3d_r50", 8),
    ("X3D_S", "X3D-S", "efficient 3D CNN", "x3d_s", 13),
    ("X3D_M", "X3D-M", "efficient 3D CNN", "x3d_m", 16),
    ("R2Plus1D_R50", "R(2+1)D R50", "factorized 3D CNN", "r2plus1d_r50", 16),
    ("C2D_R50", "C2D R50", "2D baseline CNN", "c2d_r50", 8),
    ("CSN_R101", "CSN R101", "channel-separated 3D CNN", "csn_r101", 32),
)

_HF = (
    ("TimeSformer_K400_HF", "TimeSformer Base", "divided space-time transformer", "facebook/timesformer-base-finetuned-k400", 8, "224x224", True, "TimesformerModel"),
    ("TimeSformer_HR_K400_HF", "TimeSformer HR", "divided space-time transformer", "facebook/timesformer-hr-finetuned-k400", 8, "448x448", True, "TimesformerModel"),
    ("ViViT_B_K400_HF", "ViViT-B K400", "factorized video transformer", "google/vivit-b-16x2-kinetics400", 32, "224x224", True, "VivitModel"),
    ("XCLIP_Base_P32_HF", "X-CLIP Base/P32", "video-language transformer", "microsoft/xclip-base-patch32", 8, "224x224", False, "XCLIPModel"),
    ("VideoPrism_Base_F16_HF", "VideoPrism Base F16", "video foundation encoder", "MHRDYN7/videoprism-base-f16r288", 16, "288x288", False, "VideoPrismModel"),
    ("VJEPA2_ViT_L_FPC64_HF", "V-JEPA 2 ViT-L FPC64", "self-supervised video encoder", "facebook/vjepa2-vitl-fpc64-256", 64, "256x256", False, "VJEPA2Model"),
    ("VideoMAE_Base_K400_HF", "VideoMAE Base K400 fine-tuned", "masked video autoencoder", "MCG-NJU/videomae-base-finetuned-kinetics", 16, "224x224", True, "VideoMAEModel"),
    ("VideoMAE_Large_K400_HF", "VideoMAE Large K400 fine-tuned", "masked video autoencoder", "MCG-NJU/videomae-large-finetuned-kinetics", 16, "224x224", True, "VideoMAEModel"),
)


def _build_registry() -> dict[str, ModelSpec]:
    specs: list[ModelSpec] = []
    for model_id, display, family, constructor, weights, frames, spatial in _TV:
        specs.append(ModelSpec(model_id, display, family, "torchvision", "torchvision Kinetics-400", weights, frames, spatial, True, True, True, dependencies=("torch", "torchvision"), notes=f"{constructor}; official DEFAULT weights and transforms."))
    for model_id, display, family, hub_name, frames in _PTV:
        specs.append(ModelSpec(model_id, display, family, "pytorchvideo", "PyTorchVideo Kinetics-400 hub", hub_name, frames, "224x224", True, True, True, dependencies=("torch", "pytorchvideo", "fvcore", "iopath"), notes="Official pretrained hub model; K400 normalization."))
    for model_id, display, family, source, frames, spatial, accuracy, model_class in _HF:
        specs.append(ModelSpec(model_id, display, family, "huggingface", source, source, frames, spatial, True, accuracy, True, dependencies=("torch", "transformers", "huggingface-hub", "safetensors"), option_keys=("cache_dir", "revision", "local_files_only", "token", "trust_remote_code"), notes=f"Features: {model_class} final hidden state; classifiers: AutoModelForVideoClassification."))
    specs.append(ModelSpec(
        "VideoMAE_Base_Pretrain_Local", "VideoMAE Base Pretraining", "masked video autoencoder", "videomae-local",
        "MCG-NJU/VideoMAE modeling_finetune.py", "user-supplied pretraining checkpoint", 16, "224x224", True, False, True,
        required_options=("videomae_repo", "checkpoint"), dependencies=("torch", "timm", "einops"),
        option_keys=("videomae_repo", "checkpoint"), notes="Feature-only. The repository and checkpoint are never downloaded.",
    ))
    registry = {spec.id: spec for spec in specs}
    if len(specs) != 27 or len(registry) != 27:
        raise RuntimeError("Canonical model registry must contain 27 unique endpoints")
    return registry


MODELS = _build_registry()
MODEL_IDS = tuple(MODELS)


def available_models() -> dict[str, ModelSpec]:
    """Return a shallow copy so callers cannot mutate the canonical mapping."""
    return dict(MODELS)


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_id!r}; choose from {sorted(MODELS)}") from exc


def validate_model_options(model_id: str, options: Mapping[str, Any] | None) -> dict[str, Any]:
    spec = get_model(model_id)
    if options is None:
        normalized: dict[str, Any] = {}
    elif not isinstance(options, Mapping):
        raise ValueError(f"Model options for {model_id} must be a JSON object")
    else:
        normalized = dict(options)
    unknown = sorted(set(normalized) - set(spec.option_keys))
    if unknown:
        raise ValueError(f"Unsupported options for {model_id}: {unknown}; allowed: {list(spec.option_keys)}")
    missing = [key for key in spec.required_options if not normalized.get(key)]
    if missing:
        raise ValueError(f"{model_id} requires model option(s) {missing}; pass --model-options JSON with explicit local paths")
    for key in ("model_repo", "videomae_repo", "checkpoint", "cache_dir"):
        if key in normalized and not isinstance(normalized[key], str):
            raise ValueError(f"Model option {key!r} for {model_id} must be a string path")
    if "local_files_only" in normalized and not isinstance(normalized["local_files_only"], bool):
        raise ValueError(f"Model option 'local_files_only' for {model_id} must be boolean")
    if "trust_remote_code" in normalized and not isinstance(normalized["trust_remote_code"], bool):
        raise ValueError(f"Model option 'trust_remote_code' for {model_id} must be boolean")
    return normalized
