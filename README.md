# Multi-Axis Benchmarking Code

Code-only repository for video-model task accuracy, GPU efficiency, feature and model-RDM generation, whole-clip EEG representational similarity analysis (RSA), cross-axis robustness analysis, and Pareto/RSA figures.

## Installation

Python 3.10 or newer is supported. Python 3.12 and the checked-in lock files are recommended for an exactly repeatable core environment.

With [uv](https://docs.astral.sh/uv/), `uv.lock` reproduces the complete dependency graph, including optional model backends:

```text
uv sync --extra test
uv run pytest

# Add all model runtimes when needed (large download):
uv sync --extra test --extra heavy
```

The repository's `.python-version` selects Python 3.12. The following `pip` commands are an alternative for the pinned core analysis and test environment.

PowerShell:

```powershell
Set-Location "C:\path\to\multi-axis-reproducibility"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements\test-py312.lock.txt
python -m pip install --no-deps -e .
python -m pytest
```

Bash:

```bash
cd /path/to/multi-axis-reproducibility
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/test-py312.lock.txt
python -m pip install --no-deps -e .
python -m pytest
```

The base dependencies support the dependency-free model registry, RSA, cross-axis analysis, and figures. The `heavy` extra installs the optional torchvision, PyTorchVideo, Transformers, and external-model runtime dependencies. GPU packages and model weights are platform-specific; record the resolved environment with `uv pip freeze > outputs/environment.txt` (or `python -m pip freeze` in a pip-managed environment) for every model run.

## End-To-End Run

Create a local configuration from the path-only example:

```powershell
Copy-Item configs\raw_rerun.example.json configs\raw_rerun.local.json
multi-axis-repro run --config configs\raw_rerun.local.json
```

```bash
cp configs/raw_rerun.example.json configs/raw_rerun.local.json
multi-axis-repro run --config configs/raw_rerun.local.json
```

The end-to-end command performs, for each configured model:

1. Feature extraction from uniformly sampled video frames.
2. Prefix feature extraction synchronized to every 50/100/200-ms EEG-bin endpoint. Available prefix frames are tiled to the model's full input length and concatenated across intervals before constructing each primary model RDM.
3. Classification top-1 and top-5 evaluation for accuracy-capable endpoints; feature-only endpoints are skipped.
4. Forward latency and peak-memory benchmarking.
5. Primary temporally matched, interval-integrated EEG RSA with 50, 100, and 200 ms bins, plus a secondary onset-resolved scan. The primary analysis compares concatenated EEG interval means with concatenated model-prefix features. The onset-resolved analysis vectorizes each centered five-sample EEG neighborhood and compares its RDM with one fixed full-clip RDM per model.
6. Max-statistic stimulus permutation and crossed participant/stimulus bootstrap inference.
7. Strict cross-axis assembly, robust associations, sensitivity analyses, and figures.

The command fails on missing files, unsupported models, incomplete endpoint joins, or mismatched stimulus IDs. It never falls back to bundled or synthetic study results.

All relative paths in the full-run configuration are resolved from `base_dir`, which is itself relative to the configuration file. In the supplied example, `base_dir` is `..`, so paths resolve from the repository root regardless of the shell's current directory. Relative video paths inside a clip manifest resolve from that manifest's directory.

## Individual Commands

```text
multi-axis-repro accuracy --model R3D_18 --clips data/external/classification_clips.csv --output outputs/accuracy/R3D_18.csv --device cuda

multi-axis-repro models --format markdown

multi-axis-repro models --json

multi-axis-repro efficiency --model R3D_18 --output outputs/efficiency/R3D_18.csv --device cuda --warmup 3 --repeats 10

multi-axis-repro features --model R3D_18 --clips data/external/feature_clips.csv --output outputs/features/R3D_18 --device cuda

multi-axis-repro model-rdm --features outputs/features/R3D_18/features.npy --video-ids outputs/features/R3D_18/kept_video_ids.npy --output outputs/model_rdms/R3D_18

multi-axis-repro time-synced-model-rdms --model R3D_18 --clips data/external/feature_clips.csv --timepoints-ms 50 --stimulus-duration-ms 3000 --integrated-output outputs/model_rdms/interval_integrated --integrated-bin-ms 50 100 200 --device cuda

multi-axis-repro whole-clip-rsa --eeg data/external/eeg_matrix.npy --video-ids data/external/video_ids.npy --model-rdms outputs/model_rdms/interval_integrated/100ms --output outputs/rsa/100ms --bin-ms 100 --time-start-ms -200 --time-end-ms 3400

multi-axis-repro onset-resolved-rsa --eeg data/external/eeg_matrix.npy --video-ids data/external/video_ids.npy --model-rdms outputs/model_rdms --output outputs/rsa/onset_resolved --epoch-ms -200 800 --neighborhood-samples 5 --time-start-ms -200 --time-end-ms 3400

multi-axis-repro generalization --rsa-root outputs/rsa/100ms --model-rdms outputs/model_rdms/interval_integrated/100ms --stimulus-metadata data/external/stimulus_metadata.csv --permutations 10000 --bootstraps 2000

multi-axis-repro cross-axis --manifest data/external/model_aliases.csv --task outputs/task.csv --efficiency outputs/efficiency.csv --rsa outputs/rsa/100ms/whole_clip_rsa_summary.csv --output outputs/cross_axis --bootstrap 10000 --permutations 20000

multi-axis-repro figures --endpoints outputs/cross_axis/cross_axis_endpoint_manifest.csv --output outputs/figures
```

Run `multi-axis-repro COMMAND --help` for all options.

`accuracy`, `efficiency`, and `features` accept `--model-options options.json`. This is required for local/external endpoints. In a full-run config, use `model_options` as an object keyed by model ID. See [docs/MODELS.md](docs/MODELS.md) for schemas and setup.

## External Input Schemas

These files are not included.

### Data availability

The EEG recordings cannot be distributed publicly because of institute regulations. Access to the EEG data is available upon reasonable request to the corresponding author, subject to institutional review and any applicable participant-consent or data-use conditions. All analysis code, input schemas, parameters, and random seeds needed to rerun the analyses after authorized data access are provided in this repository.

Stimulus videos, classification data, and model weights are also external inputs and remain subject to their source licenses and access terms. No restricted or participant-derived data are committed here.

### EEG

`eeg_matrix.npy` must have shape:

```text
participants x stimuli x channels x time
```

`video_ids.npy` is a one-dimensional array defining the exact stimulus order shared by EEG and model RDMs.

### Feature clips

```csv
video_id,path
```

Rows must follow `video_ids.npy` order.

### Classification clips

```csv
path,label_index
```

`label_index` must use the selected checkpoint's output-index space.

### Stimulus metadata

```csv
video_id,label
```

`label` is the action stratum used by the crossed bootstrap.

### Cross-axis aliases

```csv
endpoint_id,family,provenance,task_model,efficiency_model,rsa_model
```

Cross-axis task input requires `model,k400_top1`. Efficiency input requires `model,mean_inference_ms,peak_cuda_memory_mb`. RSA input requires `model,whole_clip_mean_r`. Names must be unique, and every alias must match exactly one row in every source table. Use registry `endpoint_id` values as stable machine keys; `task_model`, `efficiency_model`, and `rsa_model` exist only to map legacy display labels. K400 and SSv2 checkpoints are separate endpoints even when they share an architecture name.

## Supported Models

The canonical registry contains exactly 27 endpoints: 9 torchvision, 9 PyTorchVideo, 8 Hugging Face, and local VideoMAE pretraining. All support features/RDMs and efficiency; 23 have canonical K400 classifiers. The complete capability and setup table is in [docs/MODELS.md](docs/MODELS.md).

Importing or listing the registry does not import Torch and does not access the network. Official cache-backed loaders may retrieve weights only when a model execution command is run.

## Analysis Details

- Videos are decoded sequentially and frames are sampled at rounded, uniformly spaced indices over the complete decoded clip.
- Model and EEG RDMs use correlation distance and condensed upper-triangle vectors with the diagonal excluded.
- Primary EEG patterns concatenate non-overlapping temporal-bin means across channel and ordered bin. Primary model patterns concatenate the corresponding synchronized prefix feature vectors before RDM construction.
- For each synchronized primary-analysis model input, the uniformly sampled prefix available through the EEG-bin end is tiled to the endpoint's full frame count. Interval-integrated RDMs concatenate these features in EEG-bin order.
- The onset-resolved scan uses centered five-sample EEG neighborhoods without temporal averaging and compares every time-specific EEG RDM with the model RDM extracted once from the complete clip. Prefix RDMs are not used in this analysis.
- RSA uses Spearman correlation and Fisher-transformed group estimates.
- Endpoint inference uses Wilcoxon tests with Holm correction.
- Generalization inference uses structure-preserving stimulus-label permutations with maximum-statistic family-wise control.
- The crossed bootstrap resamples participants and stimuli within action labels and excludes duplicate-identity RDM pairs.
- Cross-axis analysis uses Spearman correlation, endpoint bootstrap intervals, endpoint-label permutations, Holm correction, Kendall sensitivity, leave-one-endpoint-out analysis, and descriptive leave-one-family-out analysis.
- Latency and CUDA memory are hardware- and environment-specific and should only be compared under a matched protocol.

## Repository Hygiene

The following are ignored by Git:

- `data/`
- `outputs/`
- model checkpoints and NumPy arrays
- videos
- local configuration files
- virtual environments, caches, and build metadata

Before publishing, verify with:

```powershell
git status --short
git ls-files | Select-String -Pattern '\.(npy|npz|pt|pth|ckpt|csv|png|pdf|docx)$'
```

```bash
git status --short
git ls-files | grep -E '\.(npy|npz|pt|pth|ckpt|csv|png|pdf|docx)$' || true
```

## License

Repository code is MIT licensed. Model weights, datasets, stimulus videos, and EEG recordings remain subject to their original licenses and access restrictions.
