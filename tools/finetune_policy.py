# Adapted in part from Skura3/g1-dance-rtx3050 tools/finetune_nju.py
# and tools/train_3050.py, revision 9485439831ce6a480537013a6330ab72399efe49.
# Copyright (c) 2026 G1 Dance RTX3050 contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Bounded single-clip NJU fine-tuning with an optional matched orientation tolerance."""

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def update_config(obj, values):
    allowed = {field.name for field in dataclasses.fields(obj)}
    for name, value in values.items():
        if name not in allowed:
            raise ValueError(f"Unknown configuration field {type(obj).__name__}.{name}")
        current = getattr(obj, name)
        if dataclasses.is_dataclass(current):
            update_config(current, value)
        else:
            setattr(obj, name, tuple(value) if isinstance(current, tuple) and value is not None else value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def configure_body_orientation(trainer, std):
    """Use the same orientation tolerance in real and imagined training rewards."""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("body_orientation_std must be finite and positive")
    real = trainer.env.reward_manager.get_term_cfg("motion_body_ori")
    imagined = trainer.wm.heads.reward_cfg
    before = {"simulator_std": real.params["std"], "imagination_std": imagined.std_body_ori}
    assert before["simulator_std"] == before["imagination_std"] == 0.4
    real.params["std"] = std
    imagined.std_body_ori = std
    return {"before": before, "simulator_std": real.params["std"], "imagination_std": imagined.std_body_ori,
            "positive_reward_max_unchanged": trainer.max_step_reward == imagined.max_step_reward,
            "scope": "Training rewards only; physics and the frozen physical evaluator are unchanged."}


def main(allow_fail_reset=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oldschool/ft1.json")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    import supermimic
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    settings = json.loads(args.config.read_text())
    orientation_std = float(settings.get("body_orientation_std", 0.4))
    if not math.isfinite(orientation_std) or orientation_std <= 0:
        raise ValueError("body_orientation_std must be finite and positive")
    checkpoint = ROOT / settings["checkpoint"]
    motion = ROOT / settings["motion"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig()
    update_config(cfg, state["config"])
    initial_iteration, initial_updates = int(state["iteration"]), int(state["num_updates"])
    del state
    update_config(cfg, settings["overrides"])
    from supermimic.heading_feedback import heading_feedback_dim
    cfg.policy.goal_dim = cfg.env.future_frames * 68 + (9 if cfg.env.anchor_obs else 0) + heading_feedback_dim(cfg.env.heading_obs_mode)
    cfg.policy_trainer.heading_obs_mode = cfg.env.heading_obs_mode
    cfg.run_name = settings["run_name"]
    cfg.log_root = ROOT / settings["log_root"]
    cfg.data.npz_dir = motion.parent
    cfg.data.clips = [motion.stem]
    cfg.data.library = cfg.data.manifest = cfg.data.limit = None
    cfg.resume = checkpoint
    cfg.resume_reset_sampler = True
    cfg.schedule.total_iterations = initial_iteration + settings["additional_steps"]
    assert settings["additional_steps"] > cfg.schedule.warmup_iterations >= max(cfg.wm_trainer.window + 1, cfg.policy.history)
    assert cfg.env.future_frames == cfg.policy_trainer.future_frames == 10
    assert cfg.policy.history == cfg.policy_trainer.history == 10
    assert not cfg.env.anchor_obs and not cfg.env.lock_waist
    if not allow_fail_reset:
        assert cfg.env.fail_reset_frac == 0 and cfg.env.fail_reset_list is None
    elif cfg.env.fail_reset_list is not None:
        assert (ROOT / cfg.env.fail_reset_list).is_file() or Path(cfg.env.fail_reset_list).is_file()
    with np.load(motion, allow_pickle=False) as data:
        expected_frames = int(settings.get("motion_frames", 244))
        assert expected_frames > cfg.env.future_frames
        assert float(data["fps"][0]) == 50 and data["joint_pos"].shape == (expected_frames, 29)
        assert np.isfinite(data["joint_pos"]).all()
    if args.check_config:
        print(json.dumps(dict(dataclasses.asdict(cfg), local_training_objective={"body_orientation_std": orientation_std}), indent=2, default=str))
        return
    out = cfg.log_root / cfg.run_name
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(settings, recorded_before_training_at=datetime.now(timezone.utc).isoformat(),
                checkpoint_sha256=digest(checkpoint), motion_sha256=digest(motion),
                launcher_sha256=digest(Path(__file__)), source_module=supermimic.__file__,
                initial_iteration=initial_iteration, initial_updates=initial_updates,
                effective_config=dataclasses.asdict(cfg),
                runtime={"cpu_threads": 1, "matmul_precision": "highest", "kinematics": "original upstream"})
    if cfg.env.heading_obs_mode != "off":
        source_root = Path(supermimic.__file__).resolve().parent
        source_files = ("heading_feedback.py", "config.py", "envs/tracking_env_cfg.py",
                        "envs/mdp/observations.py", "imagination/rollout.py",
                        "imagination/policy_trainer.py", "training/loop.py")
        plan["heading_source_sha256"] = {relative: digest(source_root / relative) for relative in source_files}
        plan["heading_checkpoint_adapter_sha256"] = digest(ROOT / "tools/heading_checkpoint.py")
    write_json(out / "training_plan.json", plan)
    (out / "launcher_frozen.py").write_bytes(Path(__file__).read_bytes())
    resources = {"samples": 0, "memory_peak_mib": 0.0, "temperature_max_c": 0.0}
    stop = threading.Event()
    stop_requested = False

    def monitor():
        while not stop.is_set():
            try:
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                memory, temperature, utilization = map(float, raw.strip().split(","))
                resources.update(samples=resources["samples"] + 1, current_memory_mib=memory,
                                 current_temperature_c=temperature, current_utilization_percent=utilization,
                                 memory_peak_mib=max(resources["memory_peak_mib"], memory),
                                 temperature_max_c=max(resources["temperature_max_c"], temperature))
                write_json(out / "resources.json", resources)
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            stop.wait(5.0)

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    class StopRequested(Exception):
        pass

    class LocalTrainer(Trainer):
        def load(self, path):
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            from heading_checkpoint import migrate_checkpoint
            loaded, migration = migrate_checkpoint(loaded, self.cfg.policy.goal_dim)
            networks = {}
            for name in ("wm", "policy", "critic"):
                module = getattr(self, name)
                module.load_state_dict(loaded[name], strict=True)
                networks[name] = all(torch.equal(value.detach().cpu(), loaded[name][key]) for key, value in module.state_dict().items())
            for key, optimizer in (("wm_opt", self.wm_trainer.opt), ("policy_opt", self.policy_trainer.opt), ("critic_opt", self.policy_trainer.critic_opt)):
                optimizer.load_state_dict(loaded[key])
            self.command.reset_sampler_state()
            self.iteration = int(loaded["iteration"])
            self.num_updates = int(loaded["num_updates"])
            assert all(networks.values())
            write_json(out / "resume_validation.json", {"networks_exact": networks,
                       "heading_input_migration": migration,
                       "optimizer_states_loaded": ["wm_opt", "policy_opt", "critic_opt"],
                       "sampler_reset_for_new_library": True, "replay_reinitialized": True,
                       "iteration": self.iteration, "num_updates": self.num_updates})
            print(f"Loaded original networks and optimizer states at {self.iteration}; reset sampler for one clip", flush=True)

        def save(self, name="ckpt_latest.pt"):
            for model_name in ("wm", "policy", "critic"):
                for value in getattr(self, model_name).state_dict().values():
                    if value.is_floating_point() and not torch.isfinite(value).all():
                        raise FloatingPointError(f"{model_name}: nonfinite checkpoint parameters")
            super().save(name + ".tmp")
            os.replace(self.log_dir / (name + ".tmp"), self.log_dir / name)

        def act(self, obs):
            if stop_requested:
                self.save()
                raise StopRequested("User/process stop requested; checkpoint retained")
            result = super().act(obs)
            if not torch.isfinite(result).all():
                raise FloatingPointError("Policy generated nonfinite actions")
            return result

    trainer = None
    report = {"ok": False, "plan": str(out / "training_plan.json"), "requested_additional_steps": settings["additional_steps"]}
    started = time.perf_counter()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        torch.cuda.reset_peak_memory_stats()
        trainer = LocalTrainer(cfg)
        write_json(out / "objective_contract.json", configure_body_orientation(trainer, orientation_std))
        online = trainer.wm_trainer.online_error
        trainer.wm_trainer.online_error = lambda buffer, n=1024, k=8: online(buffer, n=min(n, settings["diagnostic_batch_size"]), k=k)
        with (out / "updates.jsonl").open("x") as metrics_file:
            for component, obj in (("wm", trainer.wm_trainer), ("policy", trainer.policy_trainer)):
                original = obj.update

                def checked(*values, _fn=original, _component=component, **kwargs):
                    metrics = _fn(*values, **kwargs)
                    if not all(math.isfinite(float(value)) for value in metrics.values()):
                        raise FloatingPointError(f"{_component}: nonfinite update metrics")
                    metrics_file.write(json.dumps({"iteration": trainer.iteration, "component": _component, **metrics}) + "\n")
                    metrics_file.flush()
                    return metrics

                obj.update = checked
            try:
                trainer.run()
                report["stop_reason"] = "step_budget" if trainer.iteration == cfg.schedule.total_iterations else "time_budget"
            except StopRequested as exc:
                report["stop_reason"] = str(exc)
            report.update(ok=True, initial_iteration=initial_iteration, final_iteration=trainer.iteration,
                          additional_steps=trainer.iteration - initial_iteration,
                          optimizer_updates=trainer.num_updates - initial_updates,
                          saved_checkpoint=str(out / "ckpt_latest.pt"),
                          saved_checkpoint_sha256=digest(out / "ckpt_latest.pt"),
                          torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                          torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        if trainer is not None:
            if trainer.writer is not None:
                trainer.writer.close()
            trainer.env.close()
        report.update(wall_seconds=time.perf_counter() - started, resources=resources,
                      original_checkpoint_unchanged=digest(checkpoint) == plan["checkpoint_sha256"])
        write_json(out / "run_report.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
