"""Staged higher-hop RL (EXP-017) and direct NJU RL (EXP-018) built cleanly on finetune_policy."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_policy as base_ft
from hop_reward import (
    DIRECT_ANCHOR_OUTSIDE_S,
    DIRECT_ANCHOR_STRENGTH,
    DT,
    STAGED_ANCHOR_OUTSIDE_S,
    STAGED_ANCHOR_STRENGTH,
    DirectReward,
    HopGeometry,
    HopReward,
)

DECODER_LAST_3 = {f"decoder.mlp.net.{i}.{x}" for i in (4, 6, 8) for x in ("weight", "bias")}
REPLAY_FOCUS_S = (5.92, 6.70)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class SamplingAudit:
    """Record reset starts and per-frame transitions during training."""

    def __init__(self, trainer, clip_name):
        command = trainer.command
        env = trainer.env
        count = command.library.num_frames
        assert [x.name for x in command.library.clips] == [clip_name]
        self.starts = torch.zeros(count, dtype=torch.long, device=env.device)
        self.visits = torch.zeros_like(self.starts)
        self.ones = torch.ones_like(command.frame)
        self.steps = 0
        sample = command._sample_frames
        step = env.step

        def sampled(n):
            f = sample(n)
            self.starts.scatter_add_(0, f, self.ones[:n])
            return f

        def observed(a):
            self.visits.scatter_add_(0, command.frame, self.ones)
            self.steps += 1
            return step(a)

        command._sample_frames = sampled
        env.step = observed

    def summary(self):
        start = self.starts.tolist()
        visit = self.visits.tolist()
        return dict(
            reset_counts_by_frame=start,
            visited_goal_frames=visit,
            vector_steps=self.steps,
            total_resets=sum(start),
            pre_jump_window_resets=sum(start[281:306]),
            jump_window_visits=sum(visit[300:348]),
            total_transitions=sum(visit),
        )


def prepare_warm_start(actor_ckpt, wm_ckpt, output_ckpt):
    """Merge an actor-only imitation checkpoint into a full world-model/critic training checkpoint."""
    small = torch.load(actor_ckpt, map_location="cpu", weights_only=False)
    train = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
    assert small["policy"].keys() == train["policy"].keys()
    for name, val in small["policy"].items():
        assert val.shape == train["policy"][name].shape and torch.isfinite(val).all()
    train["policy"] = small["policy"]
    for key in ("policy_opt", "critic_opt"):
        train[key] = {"state": {}, "param_groups": train[key]["param_groups"]}
    for key in ("bin_memory", "bin_return", "bin_visits"):
        train.pop(key, None)
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train, output_ckpt)
    print(json.dumps({"warm_start": str(output_ckpt), "sha256": digest(output_ckpt)}, indent=2))


def activate_hop_trainer(mode, motion_frames=597):
    from mjlab.managers.reward_manager import RewardTermCfg
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import tracking_env_cfg
    from supermimic.envs.state_io import StateIO
    from supermimic.imagination import policy_trainer as actor_training
    from supermimic.training import loop

    reward_cls = HopReward if mode == "staged" else DirectReward
    anchor_outside = STAGED_ANCHOR_OUTSIDE_S if mode == "staged" else DIRECT_ANCHOR_OUTSIDE_S
    anchor_strength = STAGED_ANCHOR_STRENGTH if mode == "staged" else DIRECT_ANCHOR_STRENGTH

    original_config = tracking_env_cfg.make_env_cfg
    original_imagine, original_objective = actor_training.imagine, actor_training.policy_objective

    def reward_for(lib, device):
        assert len(lib.clips) == 1 and lib.clips[0].start == 0 and lib.num_frames == motion_frames
        t = torch.arange(lib.num_frames, device=device)
        qpos, qvel = lib.ref_qpos_qvel(t)
        return reward_cls(HopGeometry(load_g1_model(), device), torch.cat((qpos, qvel), -1))

    real = {}

    def in_env(env):
        cmd = env.command_manager.get_term("motion")
        if not real:
            real.update(io=StateIO(env), cost=reward_for(cmd.library, env.device))
        return real["cost"](real["io"].read(), (cmd.frame + 1).clamp_max(real["cost"].clip_last))

    def env_cfg(args):
        cfg = original_config(args)
        cfg.rewards["higher_hop_phase"] = RewardTermCfg(func=in_env, weight=1.0)
        return cfg

    tracking_env_cfg.make_env_cfg = env_cfg

    class HopTrainer(loop.Trainer):
        def __init__(self, cfg):
            super().__init__(cfg)
            for k, v in self.policy.named_parameters():
                v.requires_grad_(k in DECODER_LAST_3)
            self.anchor = copy.deepcopy(self.policy).eval().requires_grad_(False)
            self.hop = reward_for(self.lib, self.device)

            sample = self.buffer._sample_from_mask
            low, high = [round(x / DT) for x in REPLAY_FOCUS_S]

            def mixture(valid, n):
                focus = valid & (self.buffer.frame >= low) & (self.buffer.frame <= high)
                if n < 2 or not bool(focus.any()):
                    return sample(valid, n)
                ne, nt = sample(valid, n - n // 2)
                fe, ft = sample(focus, n // 2)
                return torch.cat((ne, fe)), torch.cat((nt, ft))

            self.buffer._sample_from_mask = mixture

            def imagination(wm, policy, lib, start, *args, **kwargs):
                first = []

                def capture(ob, goal):
                    a = policy(ob, goal)
                    if not first:
                        with torch.no_grad():
                            prior = self.anchor(ob.detach(), goal.detach())
                        frame_s = start.frame.to(a.dtype) * DT
                        outside = (frame_s < anchor_outside[0]) | (frame_s > anchor_outside[1])
                        per = (a - prior).square().mean(-1)
                        first.append((per * outside).sum() / outside.sum().clamp_min(1))
                    return a

                ro = original_imagine(wm, capture, lib, start, *args, **kwargs)
                ro.rewards = ro.rewards + DT * self.hop(ro.states[1:], ro.frames[1:])
                ro.actor_anchor = first[0]
                return ro

            def objective(ro, v, gamma):
                return original_objective(ro, v, gamma) - ro.rewards.shape[0] * anchor_strength * ro.actor_anchor

            actor_training.imagine = imagination
            actor_training.policy_objective = objective

    loop.Trainer = HopTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--actor", type=Path, required=True, help="Actor checkpoint (e.g. EXP-013 ckpt_actor.pt)")
    prep.add_argument("--wm-source", type=Path, required=True, help="Full checkpoint providing WM & critic (e.g. FT1 or NJU release)")
    prep.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=ROOT / "configs/oldschool/higher_hop_rl.json")
    run.add_argument("--mode", choices=("staged", "direct"), default="staged")
    run.add_argument("--check-config", action="store_true")

    args = p.parse_args()
    torch.set_num_threads(1)
    if args.action == "prepare":
        prepare_warm_start(args.actor, args.wm_source, args.output)
        return

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if not args.check_config:
        activate_hop_trainer(args.mode, int(settings.get("motion_frames", 597)))
    # Delegate config validation and training loop execution to tools/finetune_policy.py
    sys.argv = [str(ROOT / "tools/finetune_policy.py"), "--config", str(args.config)]
    if args.check_config:
        sys.argv.append("--check-config")
    base_ft.main(allow_fail_reset=True)


if __name__ == "__main__":
    main()
