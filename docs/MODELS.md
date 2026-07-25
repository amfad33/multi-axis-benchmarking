# Model Endpoints

The registry is the authoritative, dependency-free inventory. Its underscore-delimited IDs are the stable keys for all machine-readable joins; display names match the endpoint labels reported in the manuscript. Run `multi-axis-repro models --format markdown` or `multi-axis-repro models --json` without installing model backends. Listing never imports Torch or accesses the network.

| ID | Backend | Source/checkpoint | Frames | Input | Features | K400 accuracy | Efficiency | Setup |
| --- | --- | --- | ---: | --- | :---: | :---: | :---: | --- |
| `R3D_18` | torchvision | `R3D_18_Weights.DEFAULT` | 16 | 112x112 | yes | yes | yes | Official weights cache on execution |
| `MC3_18` | torchvision | `MC3_18_Weights.DEFAULT` | 16 | 112x112 | yes | yes | yes | Official weights cache on execution |
| `R2Plus1D_18` | torchvision | `R2Plus1D_18_Weights.DEFAULT` | 16 | 112x112 | yes | yes | yes | Official weights cache on execution |
| `S3D` | torchvision | `S3D_Weights.DEFAULT` | 64 | 224x224 | yes | yes | yes | Official weights cache on execution |
| `MViT_V1_B` | torchvision | `MViT_V1_B_Weights.DEFAULT` | 16 | 224x224 | yes | yes | yes | Official weights cache on execution |
| `MViT_V2_S` | torchvision | `MViT_V2_S_Weights.DEFAULT` | 16 | 224x224 | yes | yes | yes | Official weights cache on execution |
| `Swin3D_T` | torchvision | `Swin3D_T_Weights.DEFAULT` | 32 | 224x224 | yes | yes | yes | Official weights cache on execution |
| `Swin3D_S` | torchvision | `Swin3D_S_Weights.DEFAULT` | 32 | 224x224 | yes | yes | yes | Official weights cache on execution |
| `Swin3D_B` | torchvision | `Swin3D_B_Weights.DEFAULT` | 32 | 224x224 | yes | yes | yes | Official weights cache on execution |
| `SlowFast_R50` | PyTorchVideo | `slowfast_r50(pretrained=True)` | 32 | 224x224, 8+32 pathways | yes | yes | yes | PyTorchVideo hub weights on execution |
| `SlowFast_R101` | PyTorchVideo | `slowfast_r101(pretrained=True)` | 32 | 224x224, 8+32 pathways | yes | yes | yes | PyTorchVideo hub weights on execution |
| `Slow_R50` | PyTorchVideo | `slow_r50(pretrained=True)` | 8 | 224x224 | yes | yes | yes | PyTorchVideo hub weights on execution |
| `I3D_R50` | PyTorchVideo | `i3d_r50(pretrained=True)` | 8 | 224x224 | yes | yes | yes | PyTorchVideo hub weights on execution |
| `X3D_S` | PyTorchVideo | `x3d_s(pretrained=True)` | 13 | 224x224 | yes | yes | yes | PyTorchVideo hub weights on execution |
| `X3D_M` | PyTorchVideo | `x3d_m(pretrained=True)` | 16 | 224x224 | yes | yes | yes | PyTorchVideo hub weights on execution |
| `R2Plus1D_R50` | PyTorchVideo | `r2plus1d_r50(pretrained=True)` | 16 | 224x224 | yes | yes | yes | PyTorchVideo hub weights on execution |
| `C2D_R50` | PyTorchVideo | `c2d_r50(pretrained=True)` | 8 | 224x224 | yes | yes | yes | PyTorchVideo hub weights on execution |
| `CSN_R101` | PyTorchVideo | `csn_r101(pretrained=True)` | 32 | 224x224 | yes | yes | yes | PyTorchVideo hub weights on execution |
| `TimeSformer_K400_HF` | Hugging Face | `facebook/timesformer-base-finetuned-k400` | 8 | 224x224 | yes | yes | yes | Transformers model cache on execution |
| `TimeSformer_HR_K400_HF` | Hugging Face | `facebook/timesformer-hr-finetuned-k400` | 8 | 448x448 | yes | yes | yes | Transformers model cache on execution |
| `ViViT_B_K400_HF` | Hugging Face | `google/vivit-b-16x2-kinetics400` | 32 | 224x224 | yes | yes | yes | Transformers model cache on execution |
| `XCLIP_Base_P32_HF` | Hugging Face | `microsoft/xclip-base-patch32` | 8 | 224x224 | yes | no | yes | Feature-only |
| `VideoPrism_Base_F16_HF` | Hugging Face | `MHRDYN7/videoprism-base-f16r288` | 16 | 288x288 | yes | no | yes | Feature-only; recent Transformers required |
| `VJEPA2_ViT_L_FPC64_HF` | Hugging Face | `facebook/vjepa2-vitl-fpc64-256` | 64 | 256x256 | yes | no | yes | Feature-only; recent Transformers required |
| `VideoMAE_Base_K400_HF` | Hugging Face | `MCG-NJU/videomae-base-finetuned-kinetics` | 16 | 224x224 | yes | yes | yes | Transformers model cache on execution |
| `VideoMAE_Large_K400_HF` | Hugging Face | `MCG-NJU/videomae-large-finetuned-kinetics` | 16 | 224x224 | yes | yes | yes | Transformers model cache on execution |
| `VideoMAE_Base_Pretrain_Local` | local VideoMAE | user pretraining checkpoint | 16 | 224x224 | yes | no | yes | Feature-only; requires `videomae_repo`, `checkpoint` |
## Local VideoMAE

For the local VideoMAE endpoint, create an options file such as:

```json
{
  "videomae_repo": "C:/external/VideoMAE",
  "checkpoint": "C:/weights/VideoMAE_checkpoint.pth"
}
```

Use it with `--model-options path/to/options.json`. Hugging Face options may contain `cache_dir`, `revision`, `local_files_only`, `token`, and `trust_remote_code`. Full-run configuration uses a `model_options` object keyed by endpoint ID. Unknown options, missing required options, unknown endpoint keys, and options for unselected models are errors.

## Runtime Scope

All 27 endpoints support feature extraction, RDM generation from those features, and efficiency benchmarking. The four feature-only endpoints are X-CLIP, VideoPrism, V-JEPA 2 FPC64, and local VideoMAE pretraining. Individual accuracy calls reject these before loading Torch or accessing the network. Full runs skip their accuracy phase, and strict cross-axis assembly uses only configured endpoints that produced K400 accuracy rows. If a full run contains no accuracy-capable endpoint, accuracy, cross-axis assembly, and figures are omitted rather than fabricating an empty task axis.

Hugging Face feature execution uses the architecture's base model class and final hidden representation; K400 execution uses `AutoModelForVideoClassification`. The two VideoMAE endpoints mean-pool all final spatiotemporal patch tokens, matching their Hugging Face checkpoint configuration and classification implementation; VideoMAE does not prepend a class token. Torchvision and PyTorchVideo feature execution replaces the canonical classifier projection with identity. Availability still depends on compatible third-party versions and accessible weights; the repository does not claim every model/version/weight combination was runtime-tested locally.
