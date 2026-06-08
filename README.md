# GAP

本仓库包含论文 `Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation` 对应的 GAP policy 代码。该 policy 使用 DINOv3 图像 token 和 Pi3 point-map token，并通过 Point Map Prediction 辅助目标训练动作扩散策略。
源码包名统一为 `gap_policy`，对外模型类名统一为 `GAPPolicy`。

## 目录

- `scripts/process_data.py`: 从 RoboTwin 原始 HDF5 episode 中提取 DINOv3 与 Pi3 特征并写入 zarr。
- `scripts/train.py`: 训练 DINOv3 + Pi3 PMP 扩散策略。
- `deploy_policy.py`: RoboTwin 评估时加载 checkpoint 并输出动作。
- `gap_policy/`: GAP 策略网络、数据集和训练配置。
- `thirdparty/dinov3/`: DINOv3 backbone 推理所需的最小本地代码。
- `thirdparty/pi3/`: Pi3 Python 包子集；模型权重从 `pretrained/Pi3/` 加载。

## 外部文件

以下大文件不放入 git，需要在运行前放到本仓库的 `pretrained/` 目录；当前复现实验中的大小约为 Pi3 `3.6G`、DINOv3 ViT-L `1.2G`，合计约 `4.8G`。

- DINOv3 权重：`pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`。
- Pi3 权重：`pretrained/Pi3/`，即完整 Hugging Face 模型目录。

可直接运行下载脚本：

```bash
bash pretrained/download_weights.sh
```

脚本默认从 Hugging Face 下载 Pi3，并下载 GAP 依赖的 DINOv3 `.pth` 权重。若访问 gated 模型或需要更高限速，先设置 `HF_TOKEN`；若要使用自己的 DINOv3 下载地址，设置 `DINOV3_WEIGHTS_URL`。

其他外部文件：

- RoboTwin 原始数据：默认读取 `${ROBOTWIN_ROOT}/data/<task>/<task_config>/data/episode*.hdf5`；也可通过 `RAW_DATA_ROOT` 覆盖。
- 训练 checkpoint：默认写入 `checkpoints/<task>_<setting>_<expert_data_num>/<epoch>.ckpt`。

## 常用命令

在本仓库内运行预处理、训练和评估。评估时设置 `ROBOTWIN_ROOT` 指向完整 RoboTwin 根目录；GAP 会调用本仓库的 `scripts/eval_policy.py` 适配输出路径，并把当前 GAP 仓库加入 Python import 路径，不需要修改官方 RoboTwin。

```bash
cd GAP
export ROBOTWIN_ROOT=/path/to/RoboTwin
bash process_data.sh place_dual_shoes demo_clean 100 0
bash train.sh place_dual_shoes demo_clean 100 0 0 32 300 100
bash eval.sh place_dual_shoes demo_clean demo_clean 100 300 0 "0"
```

`process_data.sh` 支持通过环境变量覆盖路径：

- `ROBOTWIN_ROOT`: 完整 RoboTwin 根目录，评估必需；预处理时默认从 `${ROBOTWIN_ROOT}/data` 读原始数据。
- `RAW_DATA_ROOT`: RoboTwin 原始数据根目录，优先级高于 `ROBOTWIN_ROOT` 默认数据路径。
- `OUTPUT_ROOT`: 预处理后 zarr 输出目录，默认 `./data`。
- `GAP_PRETRAINED_ROOT`: 统一预训练权重目录，默认 `pretrained`。
- `PI3_MODEL_NAME_OR_PATH`: Pi3 本地模型目录，默认 `pretrained/Pi3`。
- `PI3_REPO`: 下载脚本使用的 Pi3 Hugging Face repo，默认 `yyfz233/Pi3`。
- `DINOV3_REPO`: 下载脚本使用的 DINOv3 `.pth` 权重 repo。
- `DINOV3_WEIGHTS_URL`: 直接指定 DINOv3 `.pth` 下载地址，优先级高于 `DINOV3_REPO`。
- `DINOV3_REPO_DIR`: DINOv3 本地代码目录，默认 `thirdparty/dinov3`。
- `DINOV3_WEIGHTS_PATH`: DINOv3 权重路径。

评估脚本会检查 `ROBOTWIN_ROOT/script/eval_policy.py` 和 checkpoint 是否存在，缺失时会直接停止。默认 checkpoint 路径为当前仓库下的 `checkpoints/<task>_<setting>_<expert_data_num>/<epoch>.ckpt`；如果这里不存在，会继续查找 `${ROBOTWIN_ROOT}/policy/GAP/checkpoints/<task>_<setting>_<expert_data_num>/<epoch>.ckpt`。也可通过 `CKPT_PATH` 指定绝对路径。
