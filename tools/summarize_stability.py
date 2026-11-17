"""Inspect absolute posture and ground contacts in paired physical trajectories."""

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def inspect_run(run):
    import mujoco
    import numpy as np

    report = json.loads((run / "report.json").read_text())
    assert len(report["clips"]) == 1 and report["resets_during_each_recorded_clip"] == 0
    clip = report["clips"][0]
    with np.load(clip["states"], allow_pickle=False) as states:
        qpos = states["qpos"].copy()
        fps = float(states["fps"][0])
    assert np.isfinite(qpos).all() and len(qpos) == clip["steps"]+1
    model = mujoco.MjModel.from_binary_path(str(run / "g1_scene.mjb"))
    data = mujoco.MjData(model)
    planes = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert len(planes) == 1, "This diagnostic expects one flat ground plane"
    names = {i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or str(i)
             for i in range(model.ngeom)}
    critical = [i for i, name in names.items()
                if name.rsplit("/", 1)[-1] in ("pelvis_collision", "torso_collision", "head_collision")]
    assert len(critical) == 3 and all(model.geom_contype[i] or model.geom_conaffinity[i] for i in critical)
    foot = {i for i, name in names.items() if re.search(r"(?:left|right)_foot[1-7]_collision$", name)}
    assert len(foot) == 14
    contacts, critical_clearance = {}, []
    floor_z = None
    for index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        floor = planes[0]
        assert np.allclose(data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1])
        floor_z = float(data.geom_xpos[floor, 2])
        clearance = []
        for geom in critical:
            radius = float(model.geom_size[geom, 0])
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_length = float(model.geom_size[geom, 1])
                radius += half_length * abs(float(data.geom_xmat[geom].reshape(3, 3)[2, 2]))
            else:
                assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
            clearance.append(float(data.geom_xpos[geom, 2]) - radius - floor_z)
        critical_clearance.append(clearance)
        touched = set()
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist > 0 or (a != floor and b != floor):
                continue
            other = b if a == floor else a
            if other not in foot:
                touched.add(names[other])
        for name in touched:
            contacts.setdefault(name, []).append(index)
    clearance = np.asarray(critical_clearance)
    quaternion = qpos[:, 3:7] / np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(1-2*(quaternion[:, 1]**2+quaternion[:, 2]**2), -1, 1)))
    touching = np.flatnonzero((clearance <= 0).any(axis=1))
    result = {
        "run": str(run.relative_to(ROOT)), "seed": report["seed"], "reset_noise": report["reset_noise"],
        "completed": clip["completed"], "failed": clip["failed"],
        "termination_flags": clip["termination_flags"], "steps": clip["steps"],
        "simulated_seconds": clip["simulated_seconds"], "saved_states": len(qpos),
        "root_height_min_m": float(qpos[:, 2].min()-floor_z),
        "root_height_max_m": float(qpos[:, 2].max()-floor_z),
        "absolute_root_tilt_max_deg": float(tilt.max()),
        "absolute_root_tilt_peak_time_s": float(tilt.argmax()/fps),
        "critical_ground_clearance_min_m": {names[geom]: float(clearance[:, j].min())
                                             for j, geom in enumerate(critical)},
        "critical_ground_touch_states": len(touching),
        "critical_first_ground_touch_time_s": float(touching[0]/fps) if len(touching) else None,
        "nonfoot_ground_contacts": {name: {"saved_states": len(indices), "first_time_s": indices[0]/fps}
                                     for name, indices in contacts.items()},
        "joint_rmse_deg": float(np.degrees(clip["joint_rmse_rad"])),
        "root_error_mean_m": clip["root_error_mean_m"],
        "yaw_error_max_abs_deg": clip["yaw_error_max_abs_deg"],
        "no_fall_indicators_in_recorded_clip": bool(clip["completed"] and not clip["failed"]
                                                    and not len(touching) and not contacts),
    }
    series = {"time": np.arange(len(qpos))/fps, "height": qpos[:, 2]-floor_z,
              "absolute_tilt": tilt, "critical_clearance": clearance.min(axis=1)}
    return result, series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-model", default="pretrained")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-model", default="ft1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, series = [], {}
    for label, path, selected_model in (("pretrained", args.baseline, args.baseline_model),
                                        ("full_ft1", args.candidate, args.candidate_model)):
        summary = json.loads(path.read_text())
        for row in summary["runs"]:
            if row["model"] != selected_model:
                continue
            values, timeline = inspect_run(path.parent / row["run"])
            values["model"] = label
            rows.append(values)
            series[(label, values["seed"])] = timeline
    candidate = [row for row in rows if row["model"] == "full_ft1"]
    assert candidate
    worst = max(candidate, key=lambda row: row["absolute_root_tilt_max_deg"])
    result = {
        "runs": rows,
        "candidate_completed_runs": sum(row["completed"] for row in candidate),
        "candidate_evaluated_runs": len(candidate),
        "candidate_no_fall_indicator_runs": sum(row["no_fall_indicators_in_recorded_clip"] for row in candidate),
        "candidate_worst_absolute_tilt_run": worst["run"],
        "ground_check": "Recompute sphere/capsule geometry clearances and ground contact overlap at every saved 50 Hz state, using each run's saved physics model. No replay integration or new policy inference.",
        "limitations": "Saved control states do not capture every 5 ms physics substep or prove contact forces. No new empirical tilt/height thresholds are used as a success gate. Completion, actual posture, ground overlap and physical video must be considered together. No long-duration, external-push or hardware validation.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    selected_seeds = [0] if worst["seed"] == 0 else [0, worst["seed"]]
    fig, axes = plt.subplots(3, len(selected_seeds), figsize=(6*len(selected_seeds), 8), squeeze=False)
    for column, seed in enumerate(selected_seeds):
        for label in ("pretrained", "full_ft1"):
            timeline = series.get((label, seed))
            if timeline is None:
                continue
            for row, key in enumerate(("height", "absolute_tilt", "critical_clearance")):
                axes[row, column].plot(timeline["time"], timeline[key], label=label)
        axes[0, column].set_title(f"Continuous physical rollout | seed {seed}")
        for row, ylabel in enumerate(("Root height (m)", "Absolute root tilt (deg)", "Minimum torso/head/pelvis clearance (m)")):
            axes[row, column].set_ylabel(ylabel)
            axes[row, column].grid(alpha=.25)
            axes[row, column].legend()
        axes[2, column].axhline(0, color="black", linewidth=.7)
        axes[2, column].set_xlabel("Elapsed simulation time (s)")
    fig.tight_layout()
    fig.savefig(output / "stability_curves.png", dpi=150)
    plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}, indent=2))
    for row in rows:
        print(json.dumps({key: row[key] for key in ("model", "seed", "completed", "root_height_min_m",
                         "absolute_root_tilt_max_deg", "critical_ground_touch_states", "nonfoot_ground_contacts")}))


if __name__ == "__main__":
    main()
