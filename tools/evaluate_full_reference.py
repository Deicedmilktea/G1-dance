"""Evaluate frozen policies on a complete reference, optionally reusing a baseline."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from record_policy_physics import orientation_quality

    settings = json.loads(args.config.read_text())
    recorder = ROOT / settings.get("recorder", "tools/record_policy_physics.py")
    heading_modes = settings.get("expected_heading_modes", {})

    def matching_contract_value(key, value):
        if key != "evaluation_env":
            return value
        value = dict(value)
        value.setdefault("heading_obs_mode", "off")
        if heading_modes:
            value.pop("heading_obs_mode")  # validated separately; the single intended A/B difference
        return value

    motion = ROOT / settings["motion_npz"]
    with np.load(motion, allow_pickle=False) as data:
        planned_frames = len(data["joint_pos"])
        fps = float(data["fps"][0])
    baseline_rows, baseline_path = {}, None
    contracts = {}
    if settings.get("baseline_summary"):
        baseline_path = ROOT / settings["baseline_summary"]
        baseline = json.loads(baseline_path.read_text())
        baseline_plan = json.loads((ROOT / baseline["plan"]).read_text())
        assert baseline["motion_sha256"] == digest(motion), "Baseline uses a different reference"
        assert baseline_plan["recorder_sha256"] == digest(recorder), "Baseline recorder changed"
        baseline_rows = {row["seed"]: row for row in baseline["runs"] if row["model"] == settings["baseline_model"]}
        expected_seeds = [settings["physical_nominal_seed"], *settings["physical_perturbed_seeds_after_nominal_completion"]]
        assert all(seed in baseline_rows for seed in expected_seeds), "Missing baseline seed"
        for seed, row in baseline_rows.items():
            contracts[seed] = json.loads((baseline_path.parent / row["run"] / "environment_contract.json").read_text())
    output = ROOT / settings["physical_evaluation_output"]
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_hashes = {name: digest(ROOT / path) for name, path in settings["policy_checkpoints"].items()}
    plan = {"recorded_before_rollouts_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
            "motion_sha256": digest(motion), "checkpoint_sha256": checkpoint_hashes,
            "recorder_sha256": digest(recorder),
            "evaluation_launcher_sha256": digest(Path(__file__)), "weight_updates": False,
            "planned_control_steps_per_complete_run": planned_frames,
            "reference_timing": "Unchanged recorder: actual post-step state vs refreshed reference. Segment masks use saved reference frame / fps.",
            "baseline_summary_sha256": digest(baseline_path) if baseline_path else None,
            "scope": settings.get("scope", "Full-reference diagnostic, no training or pilot-threshold adoption decision.")}
    save(output / "evaluation_plan.json", plan)
    (output / "launcher_frozen.py").write_bytes(Path(__file__).read_bytes())
    rows, paired_reference_runs = [], {}
    seeds = [settings["physical_nominal_seed"], *settings["physical_perturbed_seeds_after_nominal_completion"]]
    for name, checkpoint_relative in settings["policy_checkpoints"].items():
        checkpoint = ROOT / checkpoint_relative
        for ordinal, seed in enumerate(seeds):
            reset_noise = ordinal > 0
            run_name = f"nominal_seed{seed}" if not reset_noise else f"reset_noise_seed{seed}"
            run = output / name / run_name
            command = [str(ROOT / "envs/run.sh"), "training", str(recorder),
                       "--checkpoint", str(checkpoint), "--motions", str(motion), "--output", str(run), "--seed", str(seed)]
            if reset_noise:
                command.append("--reset-noise")
            run.parent.mkdir(parents=True, exist_ok=True)
            save(run.parent / f"{run_name}_command.json", {"command": command})
            print(f"Evaluating {name}: {run_name}", flush=True)
            with (run.parent / f"{run_name}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((run / "report.json").read_text())
            clip = report["clips"][0]
            assert report["checkpoint_sha256"] == checkpoint_hashes[name]
            assert report["seed"] == seed and report["reset_noise"] == reset_noise
            assert not report["weight_updates"] and report["resets_during_each_recorded_clip"] == 0
            contract = json.loads((run / "environment_contract.json").read_text())
            if heading_modes:
                assert contract["evaluation_env"]["heading_obs_mode"] == heading_modes[name]
            if seed in contracts:
                for key in ["evaluation_env", "policy_observation_shape", "goal_shape", "action_dimension",
                            "physics_dt_s", "control_dt_s", "gravity", "auto_reset", "motion_sha256"]:
                    assert matching_contract_value(key, contract[key]) == matching_contract_value(key, contracts[seed][key]), f"Model comparison changed {key}"
            else:
                contracts[seed] = contract
            with np.load(clip["states"], allow_pickle=False) as data:
                q, ref, frames = data["qpos"], data["qref"], data["frame"]
                assert q.shape == ref.shape == (clip["steps"]+1, 36)
                assert data["action"].shape == (clip["steps"], 29)
                assert all(np.isfinite(data[key]).all() for key in ["qpos", "qref", "action", "metrics"])
                if seed in baseline_rows:
                    baseline_run = baseline_path.parent / baseline_rows[seed]["run"]
                    with np.load(baseline_run / (motion.stem + "_states.npz"), allow_pickle=False) as old:
                        assert len(old["qref"]) >= len(ref), "Baseline did not reach the candidate endpoint"
                        assert np.array_equal(old["qref"][:len(ref)], ref), "Paired reference poses differ"
                        assert np.array_equal(old["frame"][:len(ref)], frames), "Paired reference timing differs"
                        assert np.array_equal(old["fps"], data["fps"])
                paired_states = None
                if seed in paired_reference_runs:
                    with np.load(paired_reference_runs[seed], allow_pickle=False) as old:
                        paired_states = min(len(old["qref"]), len(ref))
                        assert np.array_equal(old["qpos"][0], q[0]), "Paired initial physical states differ"
                        assert np.array_equal(old["qref"][:paired_states], ref[:paired_states])
                        assert np.array_equal(old["frame"][:paired_states], frames[:paired_states])
                        assert np.array_equal(old["fps"], data["fps"])
                else:
                    paired_reference_runs[seed] = Path(clip["states"])
                joint = float(np.sqrt(np.mean((q[:, 7:]-ref[:, 7:])**2)))
                root = float(np.linalg.norm(q[:, :3]-ref[:, :3], axis=-1).mean())
                assert abs(joint-clip["joint_rmse_rad"]) < 1e-7
                assert abs(root-clip["root_error_mean_m"]) < 1e-6
                segments = {}
                for segment, mask in [("prefix_before_7s", frames/fps < 7), ("tail_from_7s", frames/fps >= 7)]:
                    count = int(mask.sum())
                    value = {"recorded_states": count, "reached": count > 0}
                    if count:
                        segment_error = q[mask, 7:]-ref[mask, 7:]
                        value["joint_rmse_deg"] = float(np.rad2deg(np.sqrt(np.mean(segment_error**2))))
                        value["root_error_mean_m"] = float(np.linalg.norm(q[mask, :3]-ref[mask, :3], axis=-1).mean())
                        value.update(orientation_quality(q[mask], ref[mask], 1/fps))
                        # Segment peak times returned by the helper are relative to the selected segment.
                        value["segment_first_reference_time_s"] = float(frames[mask][0]/fps)
                    segments[segment] = value
                norm_error = max(float(np.abs(np.linalg.norm(q[:, 3:7], axis=-1)-1).max()),
                                 float(np.abs(np.linalg.norm(ref[:, 3:7], axis=-1)-1).max()))
                assert norm_error < 1e-3
            row = {"model": name, "seed": seed, "reset_noise": reset_noise,
                   "run": run.relative_to(output).as_posix(), **clip, "segments": segments,
                   "metrics_recomputed": True, "quaternion_max_norm_deviation": norm_error}
            row["paired_reference_common_states"] = paired_states
            if seed in baseline_rows:
                fields = ("completed", "failed", "steps", "simulated_seconds", "joint_rmse_rad",
                          "root_error_mean_m", "tilt_max_deg", "yaw_error_max_abs_deg")
                row["baseline"] = {key: baseline_rows[seed][key] for key in fields}
                row["baseline_run"] = str((baseline_path.parent / baseline_rows[seed]["run"]).relative_to(ROOT))
            rows.append(row)
            save(output / "partial_results.json", rows)
            print(json.dumps({"model": name, "seed": seed, "completed": clip["completed"],
                              "seconds": clip["simulated_seconds"], "flags": clip["termination_flags"],
                              "joint_rmse_deg": float(np.rad2deg(joint)), "root_mean_m": root}), flush=True)
            if ordinal == 0 and not clip["completed"]:
                print(f"{name} nominal did not complete: skip its five perturbed runs as planned.", flush=True)
                break
        assert digest(checkpoint) == checkpoint_hashes[name], "Inference changed the checkpoint"
    save(output / "summary.json", {"plan": (output / "evaluation_plan.json").relative_to(ROOT).as_posix(),
         "runs": rows, "completed_runs": sum(row["completed"] for row in rows), "evaluated_runs": len(rows),
         "motion_sha256": digest(motion), "checkpoint_sha256": checkpoint_hashes,
         "checkpoint_weights_unchanged": True, "evaluation_contracts_match_between_models": True,
         "baseline_contracts_and_reference_prefixes_match": True if baseline_rows else None,
         "weight_updates": False, "foot_sliding_quantified": False,
         "scope": settings.get("scope", "Same full reference, frozen original and pilot-B models; failed rollouts are truncated at first terminal state. Segment errors describe recorded states only, not completion of unreached portions. No pilot acceptance thresholds applied.")})


if __name__ == "__main__":
    main()
