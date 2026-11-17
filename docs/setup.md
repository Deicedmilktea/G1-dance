# 环境与模型准备

本文说明如何准备运行本仓库所需的第三方源码、Python 环境和公开模型权重。所有路径均相对于仓库根目录，权重文件和大体积缓存由 `.gitignore` 排除。

## 1. 拉取第三方源码并应用补丁

```bash
python3 tools/setup_third_party.py
```

该脚本会读取 `configs/upstream_sources.json`，将 `GVHMR`、`GMR`、`mjlab`、`pytorch3d` 和 `SuperMimic-Repro-Sim2Real` 拉取到 `third_party/` 对应固定 commit，并自动应用 `envs/motion/gmr-gvhmr-batch.patch` 和 `envs/training/heading-feedback.patch`，同时建立 `third_party/GVHMR/inputs/checkpoints` 与 `third_party/GMR/assets/body_models` 的目录映射。

环境使用方式见 [运行环境说明](../envs/README.md)。

## 2. 所需模型文件

精确文件 ID、字节数、SHA-256 校验值与存放路径保存在 [configs/models.json](../configs/models.json)。

| 文件 | 用途 | 约计大小 | 存放位置 |
| --- | --- | ---: | --- |
| `yolov8x.pt` | 检测并跟踪视频中的舞者 | 136.87 MB | `models/motion/yolo/` |
| `vitpose-h-multi-coco.pth` | 估计每帧二维人体关节 | 2.55 GB | `models/motion/vitpose/` |
| `epoch=10-step=25000.ckpt` | 提取 GVHMR 图像特征 | 2.71 GB | `models/motion/hmr2/` |
| `gvhmr_siga24_release.ckpt` | 从单目视频恢复 3D 人体动作 | 163.51 MB | `models/motion/gvhmr/` |
| `SMPLX_NEUTRAL.npz` | 人体骨架与形状模型（GVHMR 与 GMR 共用） | 108.75 MB | `models/motion/body_models/smplx/` |
| `e13i_0291000.tar.gz` | 南大 SuperMimic v1.0 预训练权重 | 266.59 MB | `data/downloads/nju/`（解压至 `models/supermimic/`） |

使用自动化脚本下载公开权重（不包含需要官网授权的 SMPL-X）：

```bash
./envs/run.sh motion tools/download_models.py
```

## 3. 手动准备 SMPL-X 模型

SMPL-X 需用户自行遵守其官方许可并下载：

1. 在 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/)注册并登录。
2. 选择 **Download SMPL-X v1.1 (NPZ+PKL)** 下载 `models_smplx_v1_1.zip`。
3. 从压缩包中提取 `SMPLX_NEUTRAL.npz`，放到 `models/motion/body_models/smplx/SMPLX_NEUTRAL.npz`。

> 注意：`.npz` 需保持为单个文件，不要再解压；不要混用 SMPL 或 SMPL-H。

## 4. 本地校验与动作恢复入口

```bash
# 检查模型文件是否存在且哈希匹配
./envs/run.sh motion tools/check_models.py --verify

# 分阶段运行人体动作恢复（默认关闭需要完整 CUDA 渲染器的 PyTorch3D 网格绘制）
./envs/run.sh motion tools/reconstruct_motion.py --video data/video/oldschool_full.mp4 --stage all --static-camera
```
