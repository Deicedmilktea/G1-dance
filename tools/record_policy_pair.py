"""Record a fixed matrix of paired, frozen policies in the same G1 simulator."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "candidate")


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def record_one(checkpoint, motion, seed, output):
    """Run one frozen policy in a fresh process and keep its log beside the trace."""
    output.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "record",
               "--checkpoint", str(checkpoint), "--motion", str(motion),
               "--output", str(output), "--seed", str(seed)]
    log_path = output.with_suffix(".log")
    try:
        with log_path.open("x", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Rollout failed; see {log_path}") from exc


def record(argv):
    import numpy as np
    import torch

    from record_policy_physics import add_sources, main as rollout

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args(argv)
    add_sources()
    import supermimic.envs as envs

    original = envs.make_env
    start, position = [], []
    max_applied = []

    def observed_env(*args, **kwargs):
        env = original(*args, **kwargs)
        action = env.action_manager.get_term("joint_pos")
        motion = env.command_manager.get_term("motion")
        assert action.process_actions.__func__.__module__ == "supermimic.envs.actions"
        assert action.noise_std == action.delay_max == motion.cfg.fail_reset_frac == 0
        assert env.physics_dt == .005 and env.step_dt == .02
        d = env.sim.data
        origin = env.scene.env_origins[0].clone()
        reset, step = env.reset, env.sim.step

        def qpos():
            q = d.qpos[0].clone()
            q[:3] -= origin
            return q

        def record_reset(*args, **kwargs):
            assert not start, "Unexpected reset within the rollout"
            out = reset(*args, **kwargs)
            start.extend((qpos(), d.qvel[0].clone()))
            return out

        def record_step():
            step()
            position.append(qpos())
            max_applied.append(d.xfrc_applied[0].abs().amax() + d.qfrc_applied[0].abs().amax())

        env.reset, env.sim.step = record_reset, record_step
        return env

    envs.make_env = observed_env
    sys.argv = ["record_policy_physics.py", "--checkpoint", str(a.checkpoint),
                "--motions", str(a.motion), "--output", str(a.output), "--seed", str(a.seed)]
    if a.seed:
        sys.argv.append("--reset-noise")
    try:
        rollout()
        assert len(start) == 2 and position
        np.savez_compressed(a.output / "physics_trace.npz",
                            initial_qpos=start[0].cpu().numpy(), initial_qvel=start[1].cpu().numpy(),
                            physics_qpos=torch.stack(position).cpu().numpy(),
                            external_force_max=torch.stack(max_applied).cpu().numpy())
    finally:
        envs.make_env = original


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--motions", nargs="+", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True,
                   help="0 uses a fixed start; each positive seed uses the upstream reset perturbations")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    motions = [x.resolve() for x in a.motions]
    if len({x.stem for x in motions}) != len(motions):
        p.error("Motion file stems must be distinct")
    if len(set(a.seeds)) != len(a.seeds) or min(a.seeds) < 0:
        p.error("Seeds must be distinct nonnegative integers")
    cp = {k: getattr(a, k).resolve() for k in ARMS}
    digests = {x: sha256(x) for x in [*cp.values(), *motions]}
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", {
        "policy_sha256": {k: digests[v] for k, v in cp.items()},
        "motion_sha256": {v.stem: digests[v] for v in motions},
        "seeds": a.seeds, "reset_noise": "off for seed 0; otherwise upstream perturbations",
        "weight_updates": False, "action_compensation": False,
    })
    for motion in motions:
        for seed in a.seeds:
            for arm, checkpoint in cp.items():
                run = output / motion.stem / f"seed{seed}" / arm
                print(f"{motion.stem} | start {seed} | {arm}", flush=True)
                record_one(checkpoint, motion, seed, run)
                report = json.loads((run / "report.json").read_text())
                contract = json.loads((run / "environment_contract.json").read_text())
                assert report["checkpoint_sha256"] == digests[checkpoint]
                assert contract["motion_sha256"] == {str(motion): digests[motion]}
    assert all(sha256(x) == h for x, h in digests.items()), "An input file changed"
    write_json(output / "finished.json", {"input_hashes_unchanged": True})


if __name__ == "__main__":
    if sys.argv[1:2] == ["record"]:
        record(sys.argv[2:])
    else:
        main(sys.argv[1:])
