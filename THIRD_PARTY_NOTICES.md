# Third-Party Notices

This repository's original code, configurations and documentation are released under the MIT License (see [`LICENSE`](LICENSE)). Portions of the training and rollout tooling build on or adapt the upstream open-source projects below; all upstream copyrights and licenses remain with their respective authors.

| Project | Role in G1-dance | License |
|---|---|---|
| [SuperMimic-Repro-Sim2Real](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real) | Pretrained G1 policy (`e13i_0291000`), motion conversion & SuperMimic training stack | MIT (Copyright (c) 2026 Yuxi Liu, Hongyu Ding) |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | Underlying SuperMimic world-model & BPTT actor-critic implementation | MIT (Copyright (c) 2026 Yuxi Liu) |
| [Skura3/g1-dance-rtx3050](https://github.com/Skura3/g1-dance-rtx3050) | Adapted bounded fine-tuning launcher and policy rollout recorder | MIT (Copyright (c) 2026 G1 Dance RTX3050 contributors) |
| [GVHMR](https://github.com/zju3dv/GVHMR) | Monocular 3D human motion recovery from video | MIT / upstream model weights terms |
| [GMR](https://github.com/YanjieZe/GMR) | Human-to-Unitree-G1 motion retargeting | MIT |
| [mjlab](https://github.com/mujocolab/mjlab) | Simulation environment framework & Unitree G1 MJCF assets | Apache-2.0 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Physics simulation | Apache-2.0 |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | Pure-Python rotation transforms subset | BSD-3-Clause |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | Human body model (`SMPLX_NEUTRAL.npz`, obtained separately by the user) | SMPL-X Body Model License |

## Reference Choreography

The demonstration video uses Cai Xukun's official *Old School* dance video strictly for non-commercial research/reproduction reference. Raw source videos, audio tracks, SMPL-X files and downloaded model checkpoints are excluded from Git and must be prepared locally by the user.
