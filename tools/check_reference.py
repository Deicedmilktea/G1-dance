"""Check a configured G1 reference against the training robot and its source timeline."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oldschool/pipeline.json")
    args = parser.parse_args()
    import mujoco
    import numpy as np
    sys.path.insert(0, str(ROOT / "third_party/SuperMimic-Repro-Sim2Real/tools/video2motion"))
    import g1_model

    settings = json.loads(args.config.read_text())
    path = ROOT / settings["reference_csv"]
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    assert rows.shape == (settings["frames"], 36) and np.isfinite(rows).all()
    model = g1_model.load_g1(str(ROOT / settings["robot_xml"]))
    lower, upper = g1_model.joint_limits(model)
    violations = (rows[:, 7:] < lower-1e-6) | (rows[:, 7:] > upper+1e-6)
    assert not violations.any(), "Reference exceeds the actual robot joint ranges"
    quaternion_error = float(np.abs(np.linalg.norm(rows[:, 3:7], axis=-1)-1).max())
    assert quaternion_error < 1e-5
    fps = settings["source_fps"]
    speed = np.abs(np.diff(rows[:, 7:], axis=0)) * fps
    top = np.argsort(speed.ravel())[-12:][::-1]
    transitions = []
    for index in top:
        frame, joint = np.unravel_index(index, speed.shape)
        transitions.append({"joint": g1_model.G1_JOINT_NAMES[joint], "from_frame": int(frame),
                            "to_frame": int(frame+1), "time_s": float((frame+1)/fps),
                            "speed_rad_s": float(speed[frame, joint])})
    data, feet = mujoco.MjData(model), g1_model.foot_contact_points(model)
    sole, feet_xy, tilt = [], [], []
    for row in rows:
        g1_model.set_row(model, data, row)
        sole.append(g1_model.sole_z(data, feet))
        feet_xy.append([data.xpos[body, :2].copy() for body, _, _ in feet])
        q = data.qpos[3:7]
        tilt.append(float(np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1)))))
    sole, feet_xy = np.array(sole), np.array(feet_xy)
    foot_speed = np.linalg.norm(np.diff(feet_xy, axis=0), axis=-1)*fps
    proximity = (sole[1:] < .02) & (sole[:-1] < .02)
    proxy = {}
    for j, name in enumerate(g1_model.FOOT_BODIES):
        values = foot_speed[:, j][proximity[:, j]]
        proxy[name] = {"near_floor_transitions": len(values),
                       "mean_horizontal_link_speed_m_s": float(values.mean()) if len(values) else None,
                       "max_horizontal_link_speed_m_s": float(values.max()) if len(values) else None}
    result = {"frames": len(rows), "fps": fps, "csv_columns": 36, "all_finite": True,
              "source_last_sample_s": (len(rows)-1)/fps, "display_duration_s": len(rows)/fps,
              "joint_limit_violations": int(violations.sum()), "quaternion_norm_max_error": quaternion_error,
              "max_joint_speed_rad_s": transitions[0]["speed_rad_s"], "fastest_joint_transitions": transitions,
              "min_sole_z_m": float(sole.min()), "max_lowest_sole_z_m": float(sole.min(axis=1).max()),
              "max_root_tilt_deg": max(tilt), "root_z_range_m": [float(rows[:, 2].min()), float(rows[:, 2].max())],
              "max_adjacent_root_translation_m": float(np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=-1).max()),
              "near_floor_foot_motion_proxy": proxy,
              "proxy_note": "Foot-link xy speed when the sole is below 2 cm in both neighboring poses; proximity only, no contact forces or proven stance phase.",
              "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "robot_xml_sha256": hashlib.sha256((ROOT / settings["robot_xml"]).read_bytes()).hexdigest(),
              "scope": "Kinematic reference checks only; speed and near-floor movement are diagnostics, not physical feasibility tests."}
    with (path.parent / "reference_check.json").open("x") as f:
        f.write(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(path.parent / "reference_geometry.npz", sole_z=sole, foot_xy=feet_xy, root_tilt_deg=np.array(tilt))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
