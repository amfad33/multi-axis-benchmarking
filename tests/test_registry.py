import builtins
import json
import sys

import pytest

from multi_axis_repro import cli
from multi_axis_repro.registry import MODEL_IDS, MODELS, available_models, validate_model_options


CANONICAL_IDS = {
    "R3D_18", "MC3_18", "R2Plus1D_18", "S3D", "MViT_V1_B", "MViT_V2_S",
    "Swin3D_T", "Swin3D_S", "Swin3D_B", "SlowFast_R50", "SlowFast_R101",
    "Slow_R50", "I3D_R50", "X3D_S", "X3D_M", "R2Plus1D_R50", "C2D_R50",
    "CSN_R101", "TimeSformer_K400_HF", "TimeSformer_HR_K400_HF",
    "ViViT_B_K400_HF", "XCLIP_Base_P32_HF", "VideoPrism_Base_F16_HF",
    "VJEPA2_ViT_L_FPC64_HF", "VideoMAE_Base_K400_HF", "VideoMAE_Large_K400_HF",
    "VideoMAE_Base_Pretrain_Local",
}


def test_registry_is_exactly_the_canonical_27():
    assert len(MODELS) == len(MODEL_IDS) == len(set(MODEL_IDS)) == 27
    assert set(MODELS) == CANONICAL_IDS
    assert set(available_models()) == CANONICAL_IDS
    assert all(spec.id == model_id for model_id, spec in MODELS.items())


def test_ambiguous_endpoints_use_manuscript_display_names():
    assert MODELS["I3D_R50"].display == "I3D-R50"
    assert MODELS["TimeSformer_K400_HF"].display == "TimeSformer Base"
    assert MODELS["TimeSformer_HR_K400_HF"].display == "TimeSformer HR"
    assert MODELS["VideoMAE_Base_Pretrain_Local"].display == "VideoMAE Base Pretraining"
    assert MODELS["VideoMAE_Base_K400_HF"].display == "VideoMAE Base K400 fine-tuned"
    assert MODELS["VideoMAE_Large_K400_HF"].display == "VideoMAE Large K400 fine-tuned"


def test_registry_capabilities_and_backend_counts():
    assert sum(spec.features for spec in MODELS.values()) == 27
    assert sum(spec.k400_accuracy for spec in MODELS.values()) == 23
    assert sum(spec.efficiency for spec in MODELS.values()) == 27
    assert {spec.id for spec in MODELS.values() if not spec.k400_accuracy} == {
        "XCLIP_Base_P32_HF", "VideoPrism_Base_F16_HF",
        "VJEPA2_ViT_L_FPC64_HF", "VideoMAE_Base_Pretrain_Local",
    }
    assert {backend: sum(spec.backend == backend for spec in MODELS.values()) for backend in {spec.backend for spec in MODELS.values()}} == {
        "torchvision": 9, "pytorchvideo": 9, "huggingface": 8, "videomae-local": 1,
    }


def test_model_options_are_strict_and_required():
    assert validate_model_options("R3D_18", None) == {}
    assert validate_model_options("TimeSformer_K400_HF", {"local_files_only": True}) == {"local_files_only": True}
    with pytest.raises(ValueError, match="Unsupported options"):
        validate_model_options("R3D_18", {"checkpoint": "weights.pth"})
    with pytest.raises(ValueError, match="requires model option"):
        validate_model_options("VideoMAE_Base_Pretrain_Local", {})
    with pytest.raises(ValueError, match="must be boolean"):
        validate_model_options("XCLIP_Base_P32_HF", {"local_files_only": "yes"})
    with pytest.raises(ValueError, match="JSON object"):
        validate_model_options("R3D_18", [])


def test_cli_models_json_lists_all_endpoints_without_torch(monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("models listing imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["multi-axis-repro", "models", "--json"])
    cli.main()
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 27
    assert {row["id"] for row in rows} == CANONICAL_IDS


def test_feature_only_accuracy_rejected_before_heavy_import(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "multi_axis_repro.heavy"} or name.startswith("torch."):
            raise AssertionError("accuracy rejection reached a heavy import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", [
        "multi-axis-repro", "accuracy", "--model", "XCLIP_Base_P32_HF",
        "--clips", str(tmp_path / "clips.csv"), "--output", str(tmp_path / "result.csv"),
    ])
    with pytest.raises(ValueError, match="feature-only"):
        cli.main()


def test_full_config_validates_per_model_options(tmp_path):
    config = {
        "eeg_matrix": "eeg.npy", "video_ids": "ids.npy", "feature_manifest": "clips.csv",
        "stimulus_metadata": "stimuli.csv", "model_rdm_root": "rdms",
        "classification_manifest": "classification.csv", "alias_manifest": "aliases.csv",
        "output_root": "outputs", "models": ["VideoMAE_Base_Pretrain_Local"],
        "model_options": {"VideoMAE_Base_Pretrain_Local": {"videomae_repo": "repo", "checkpoint": "model.pth"}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded, options = cli._full_config(path)
    assert loaded["models"] == ["VideoMAE_Base_Pretrain_Local"]
    assert loaded["eeg_matrix"] == str((tmp_path / "eeg.npy").resolve())
    assert options["VideoMAE_Base_Pretrain_Local"]["checkpoint"] == str((tmp_path / "model.pth").resolve())


def test_full_config_base_dir_is_relative_to_config(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config = {
        "base_dir": "..", "eeg_matrix": "data/eeg.npy", "video_ids": "data/ids.npy",
        "feature_manifest": "data/clips.csv", "stimulus_metadata": "data/stimuli.csv",
        "model_rdm_root": "outputs/rdms", "classification_manifest": "data/classification.csv",
        "alias_manifest": "data/aliases.csv", "output_root": "outputs", "models": ["R3D_18"],
    }
    path = config_dir / "run.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded, _ = cli._full_config(path)
    assert loaded["eeg_matrix"] == str((project / "data" / "eeg.npy").resolve())
