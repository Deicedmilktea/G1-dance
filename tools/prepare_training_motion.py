"""Convert a retargeted G1 CSV to the 50 Hz SuperMimic training NPZ format."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oldschool/pipeline.json")
    parser.add_argument("--csv", type=Path, help="Override input retargeted CSV path")
    parser.add_argument("--output", type=Path, help="Override output 50 Hz NPZ path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import numpy as np
    from supermimic.data.motion_convert import G1_JOINT_NAMES, convert_csv_to_npz, g1_xml_path

    plan = json.loads(args.config.read_text(encoding="utf-8"))
    source = (args.csv or ROOT / plan["reference_csv"]).resolve()
    output = (args.output or ROOT / plan["motion_npz"]).resolve()
    input_fps = float(plan.get("source_fps", 30))
    control_fps = float(plan.get("control_fps", 50))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Preserve existing converted motion: {output}")
    summary = convert_csv_to_npz(source, output, input_fps, control_fps)
    with np.load(output, allow_pickle=False) as motion:
        count = len(motion["joint_pos"])
        assert tuple(motion["joint_names"].tolist()) == G1_JOINT_NAMES
        assert motion["joint_pos"].shape == motion["joint_vel"].shape == (count, 29)
        assert motion["body_pos_w"].shape == (count, 30, 3)
        assert motion["body_quat_w"].shape == (count, 30, 4)
        assert float(motion["fps"][0]) == control_fps
        for name in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
            assert np.isfinite(motion[name]).all(), name
        assert np.max(np.abs(np.linalg.norm(motion["body_quat_w"], axis=-1) - 1)) < 1e-5
    summary.update({
        "input_fps": input_fps,
        "output_fps": control_fps,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "robot_xml": str(g1_xml_path()),
        "robot_xml_sha256": hashlib.sha256(g1_xml_path().read_bytes()).hexdigest(),
        "input_last_sample_s": (summary["input_frames"] - 1) / input_fps,
        "endpoint_note": "Resampling uses sample times, not displayed video duration; the final sample falls on the 50 Hz grid.",
        "all_numeric_arrays_finite": True,
        "scope": "Reference conversion and forward kinematics only; no learned-policy execution.",
    })
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
