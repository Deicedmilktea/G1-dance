# 运行环境说明

项目分成两套隔离的 Python 环境，统一通过根目录入口 `./envs/run.sh motion ...` 或 `./envs/run.sh training ...` 调用（所有环境变量、缓存路径与 `G1_PROJECT_ROOT` 均自动配置在仓库 `.runtime/` 内）：

| 用途 | 环境目录 | Python | 主要依赖 |
| --- | --- | --- | --- |
| 视频预处理、GVHMR 动作恢复、GMR 重定向 | `.venv-motion/` | 3.10 | Torch 2.5.1+cu121、MuJoCo、GVHMR、GMR、PyTorch3D（纯 Python transforms） |
| SuperMimic 策略训练、模仿学习与物理评测 | `envs/training/.venv/` | 3.12 | Torch 2.14.1+cu126、Warp 1.18.0、MuJoCo 3.16.0、mjlab、SuperMimic |

## 第三方源码与补丁

本仓库不直接提交上游第三方源码（见 `.gitignore`）。`configs/upstream_sources.json` 固定记录了五个上游仓库的 commit/版本及本地适配补丁：

1. 使用辅助脚本拉取固定版本源码并自动应用补丁：
   ```bash
   python3 tools/setup_third_party.py
   ```
2. 包含两项主要源码补丁：
   - `envs/motion/gmr-gvhmr-batch.patch`：为 GMR 加载 GVHMR 序列显式传入 `num_betas` 与序列 `batch_size`。
   - `envs/training/heading-feedback.patch`：为 `SuperMimic-Repro-Sim2Real` 增加可选 `heading_obs_mode` 开关（Oldschool 主线全部保持 `"off"`）。
   - `envs/training/nju-local-sources.patch`：仅作历史记录，用于将上游 `pyproject.toml` 中的 `mjlab` 指向本地 `third_party/mjlab`。

## 依赖配置

- 训练环境声明位于 `envs/training/pyproject.toml` 与 `envs/training/uv.lock`，在克隆后执行：
  ```bash
  uv sync --project envs/training
  ```
- 如果本机离线缓存了 Torch/Warp wheel，可自行在本地覆盖源配置，请勿提交含本机绝对路径的文件。
- `envs/motion/installed.txt` 与 `envs/training/installed.txt` 记录了当前环境包版本快照供排查参考。
