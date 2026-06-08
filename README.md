# GAP: Action-Geometry Prediction

Official implementation of **Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation**, accepted to **CVPR 2026**.

[![arXiv](https://img.shields.io/badge/arXiv-2602.23814-b31b1b.svg)](https://arxiv.org/abs/2602.23814)
[![Project](https://img.shields.io/badge/Homepage-Chongyang%20Xu-blue)](https://chongyang-99.github.io/)
[![Code](https://img.shields.io/badge/Code-GAP-black?logo=github)](https://github.com/Chongyang-99/GAP)

**[Chongyang Xu](https://chongyang-99.github.io/), [Haipeng Li](https://lhaippp.github.io/), [Shen Cheng](https://scholar.google.com/citations?user=gBE3HvwAAAAJ&hl=en), [Haoqiang Fan](https://scholar.google.com/citations?user=bzzBut4AAAAJ&hl=en), [Ziliang Feng](https://cs.scu.edu.cn/info/1279/13685.htm), [Shuaicheng Liu](http://www.liushuaicheng.org/)<sup>†</sup>**

<sup>†</sup> Corresponding author.

## News

- **2026-06-08:** GAP code, preprocessing scripts, training pipeline, and RoboTwin evaluation wrapper are released.
- **2026-02-27:** The paper is available on [arXiv](https://arxiv.org/abs/2602.23814).
- **2026-02:** GAP is accepted to **CVPR 2026**.

GAP is a bimanual manipulation policy that reasons over RGB observations with a 3D geometric prior. The policy fuses DINOv3 visual tokens, Pi3 geometry-aware point-map tokens, and robot proprioception, then uses a diffusion decoder to predict both the next action chunk and a future 3D latent target. This action-geometry prediction objective encourages the policy to understand where the scene is going, not only which motor command should come next.

## Highlights

- **RGB-only 3D reasoning.** GAP uses a pretrained 3D geometric foundation model to obtain geometry-aware latents without requiring explicit depth sensors or point-cloud inputs.
- **Joint action and geometry prediction.** The policy predicts future actions together with future 3D point-map latents, providing an auxiliary objective aligned with bimanual manipulation.
- **RoboTwin-compatible release.** The repository keeps RoboTwin as an external dependency and runs evaluation through a local GAP wrapper, so users do not need to modify the official RoboTwin codebase.
- **Reproducible scripts.** Data preprocessing, training, checkpoint loading, and evaluation are exposed through short shell entry points.

## Repository Layout

```text
GAP/
├── deploy_policy.py              # Policy wrapper used by RoboTwin evaluation
├── deploy_policy.yml             # Default evaluation config
├── process_data.sh               # Feature extraction entry point
├── train.sh                      # Training entry point
├── eval.sh                       # RoboTwin evaluation entry point
├── gap_policy/                   # GAP policy, dataset, diffusion modules, configs
├── scripts/
│   ├── process_data.py           # HDF5 episodes -> GAP zarr features
│   ├── train.py                  # Hydra training script
│   └── eval_policy.py            # GAP-local copy of RoboTwin eval runner
├── thirdparty/
│   ├── dinov3/                   # DINOv3 inference code used by GAP
│   └── pi3/                      # Pi3 model code subset used by GAP
└── pretrained/
    └── download_weights.sh       # Downloads external pretrained weights
```

The `thirdparty/` directory contains the code needed for inference. Large pretrained weights are not tracked by git.

## Prerequisites

GAP is designed to run inside a working RoboTwin environment. Please first install RoboTwin following its official instructions and verify that the simulator can run a standard policy evaluation.

After activating the RoboTwin environment, clone this repository:

```bash
git clone https://github.com/Chongyang-99/GAP.git
cd GAP
export ROBOTWIN_ROOT=/path/to/RoboTwin
```

`ROBOTWIN_ROOT` must point to the complete RoboTwin root directory and contain `script/eval_policy.py`.

## Pretrained Weights

GAP expects all pretrained weights under `pretrained/`:

```text
pretrained/
├── Pi3/
│   ├── config.json
│   └── model.safetensors or pytorch_model.bin
└── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Download them with:

```bash
bash pretrained/download_weights.sh
```

The script downloads Pi3 from Hugging Face and the DINOv3 ViT-L checkpoint used by GAP. If Hugging Face access requires authentication, set `HF_TOKEN` before running the script. The expected weight size is about 4.8 GB in total, so these files are intentionally ignored by git.

Useful overrides:

```bash
export GAP_PRETRAINED_ROOT=/path/to/pretrained
export PI3_MODEL_NAME_OR_PATH=/path/to/pretrained/Pi3
export DINOV3_WEIGHTS_PATH=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
export DINOV3_REPO_DIR=/path/to/GAP/thirdparty/dinov3
```

## Data Preparation

GAP reads RoboTwin demonstration episodes from:

```text
${ROBOTWIN_ROOT}/data/<task_name>/<task_config>/data/episode*.hdf5
```

For the demo-clean setting used in our experiments, preprocess one task with:

```bash
bash process_data.sh place_dual_shoes demo_clean 100 0
```

Arguments:

```text
process_data.sh <task_name> <task_config> <expert_data_num> <gpu_id>
```

The script extracts DINOv3 and Pi3 features and saves a zarr dataset to:

```text
data/<task_name>-<task_config>-<expert_data_num>-pi3-20-5.zarr
```

You can redirect input or output locations without modifying RoboTwin:

```bash
export RAW_DATA_ROOT=/path/to/robotwin/data
export OUTPUT_ROOT=/path/to/gap/data
```

## Training

Train GAP on the preprocessed demonstrations:

```bash
bash train.sh place_dual_shoes demo_clean 100 0 0 32 300 100
```

Arguments:

```text
train.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id> <batch_size> <num_epochs> <checkpoint_every>
```

Checkpoints are saved to:

```text
checkpoints/<task_name>_<task_config>_<expert_data_num>/<epoch>.ckpt
```

By default, logging uses offline Weights & Biases mode. To change it:

```bash
export WANDB_MODE=online
```

## Evaluation

Evaluate a trained checkpoint in RoboTwin:

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
bash eval.sh place_dual_shoes demo_clean demo_clean 100 100 0 "0"
```

Arguments:

```text
eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <checkpoint_num> <gpu_id> <seeds>
```

The default checkpoint path is:

```text
checkpoints/<task_name>_<ckpt_setting>_<expert_data_num>/<checkpoint_num>.ckpt
```

To evaluate a specific checkpoint:

```bash
export CKPT_PATH=/path/to/checkpoint.ckpt
bash eval.sh place_dual_shoes demo_clean demo_clean 100 100 0 "0"
```

Evaluation results are written inside this GAP repository:

```text
results/<task_name>/GAP/<task_config>/<ckpt_setting>/seed_<seed>/<checkpoint_num>/_result.txt
```

Set `RESULTS_ROOT` to use another output directory:

```bash
export RESULTS_ROOT=/path/to/results
```

## Configuration

The main training config is:

```text
gap_policy/config/GAP.yaml
```

Important options:

- `policy.dinov3_repo_dir`: local DINOv3 code path.
- `policy.dinov3_weights_path`: DINOv3 checkpoint path.
- `policy.pi3_model_name_or_path`: Pi3 checkpoint directory.
- `observation_chunk`: temporal observation window, default `20`.
- `interval`: temporal sampling interval, default `5`.
- `model_3d`: 3D backbone name, currently `pi3`.

## Notes on RoboTwin Integration

GAP intentionally treats RoboTwin as an external environment:

- preprocessing reads official RoboTwin HDF5 demonstrations;
- evaluation changes into `ROBOTWIN_ROOT` so RoboTwin task configs and simulator imports resolve normally;
- GAP uses `scripts/eval_policy.py` in this repository to support repository-local result paths;
- no files inside the official RoboTwin repository need to be edited.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{xu2026gap,
  title={Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation},
  author={Xu, Chongyang and Li, Haipeng and Cheng, Shen and Fan, Haoqiang and Feng, Ziliang and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [Pi3](https://github.com/yyfz/Pi3), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), and Xu et al.'s [Diffusion-Based Imaginative Coordination](https://github.com/return-sleep/Diffusion_based_imaginative_Coordination) repository for releasing their excellent codebases. GAP builds on RoboTwin for bimanual manipulation evaluation, uses Pi3 for 3D geometric representation, and benefits from Xu et al.'s open-source bimanual policy implementation.
