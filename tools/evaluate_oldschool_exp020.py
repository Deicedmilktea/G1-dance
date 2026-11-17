"""Run the locked Oldschool comparison or summarize its checked-in metrics."""

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean

from record_policy_pair import ARMS, ROOT, record_one, sha256, write_json

MANIFEST = ROOT / "configs/oldschool/exp020.json"
RESULTS = ROOT / "results/oldschool/exp020.json"
REFERENCES = ("original", "high")
FLAGS = ("completed", "completed_without_fall", "sustained_jump", "jump_and_stable_landing")
METRICS = (
    "lower_foot_peak_cm", "both_foot_vertical_rmse_cm", "left_knee_rmse_deg",
    "right_knee_rmse_deg", "ankle_width_rmse_cm", "pelvis_position_mean_cm",
    "outside_jump_mean_pelvis_position_cm", "whole_joint_rmse_deg", "arms_joint_rmse_deg",
    "outside_jump_joint_rmse_deg", "tail_mean_abs_yaw_deg", "terminal_abs_yaw_deg",
    "landing_tilt_deg", "contact_free_span_s",
)
HEADLINE = (
    ("完成舞段且无非脚部触地", "completed_without_fall", None),
    ("双脚腾空并稳定落地", "jump_and_stable_landing", None),
    ("较低脚底峰值净空", "lower_foot_peak_cm", "cm"),
    ("双脚竖直轨迹 RMSE", "both_foot_vertical_rmse_cm", "cm"),
    ("骨盆平均位置误差", "pelvis_position_mean_cm", "cm"),
    ("末帧绝对朝向误差", "terminal_abs_yaw_deg", "°"),
    ("全片关节角 RMSE", "whole_joint_rmse_deg", "°"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def input_digests(protocol):
    return {key: spec["sha256"] for key, spec in protocol["inputs"].items()}


def seed_order(protocol):
    return [protocol["display_seed"], *protocol["held_out_seeds"]]


def comparison(protocol, rows):
    return {"experiment": protocol["experiment"], "input_sha256": input_digests(protocol),
            "references": rows}


def validate_report(data, protocol):
    if (data["experiment"] != protocol["experiment"] or
            data["input_sha256"] != input_digests(protocol) or
            set(data["references"]) != set(REFERENCES)):
        raise ValueError("Report does not match the locked EXP-020 inputs")
    for ref in REFERENCES:
        rows = data["references"][ref]
        seeds = [row["seed"] for row in rows]
        if (not seeds or len(set(seeds)) != len(seeds) or
                not set(seeds) <= set(seed_order(protocol)) or
                seeds != [r["seed"] for r in data["references"][REFERENCES[0]]]):
            raise ValueError(f"Inconsistent initial-state pairing in {ref}")
        for row in rows:
            for arm in ARMS:
                values = row[arm]
                for key in FLAGS:
                    if type(values[key]) is not bool:
                        raise ValueError(f"Expected a boolean in {ref}/{row['seed']}/{arm}/{key}")
                for key in METRICS:
                    value = values[key]
                    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                        raise ValueError(f"Expected a finite metric in {ref}/{row['seed']}/{arm}/{key}")


def aggregate(rows):
    paired = [row for row in rows if all(row[arm]["completed"] for arm in ARMS)]
    counts = {arm: {k: sum(row[arm][k] for row in rows) for k in FLAGS} for arm in ARMS}
    means = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for key in METRICS:
            values = [r[arm][key] for r in paired if r[arm][key] is not None]
            means[arm][key] = {"n": len(values), "mean": mean(values) if values else None}
    return {"n": len(rows), "n_jointly_completed": len(paired), "counts": counts, "means": means}


def markdown_report(data, protocol, *, include_display=False):
    validate_report(data, protocol)
    lines = []
    for ref in REFERENCES:
        rows = [r for r in data["references"][ref] if include_display or r["seed"] != protocol["display_seed"]]
        lines.append("### " + ("同一原始 G1 参考" if ref == "original" else "同一修订高跳参考"))
        if not rows:
            lines.extend(("", "本次只评测了展示初态。用 `--include-display` 查看。", ""))
            continue
        stats = aggregate(rows)
        seeds = ", ".join(str(row["seed"]) for row in rows)
        lines.extend(("", f"初态：{seeds}。共同完成：{stats['n_jointly_completed']}/{stats['n']}。", "",
                      "| 指标 | 南大发布策略 | 成片 RL |", "| --- | ---: | ---: |"))
        for label, key, unit in HEADLINE:
            if unit is None:
                cells = [f"{stats['counts'][arm][key]}/{stats['n']}" for arm in ARMS]
            else:
                cells = ["—" if (v := stats["means"][arm][key]["mean"]) is None
                         else f"{v:.2f}{unit if unit == '°' else ' ' + unit}" for arm in ARMS]
            lines.append(f"| {label} | {' | '.join(cells)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def pose_file(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as motion:
        pelvis = motion["body_names"].tolist().index("pelvis")
        q = np.c_[motion["body_pos_w"][:, pelvis], motion["body_quat_w"][:, pelvis],
                  motion["joint_pos"]].astype(float)
        joints = motion["joint_names"]
    return q, joints


def normalized_contract(contract):
    # The recorder logs absolute input paths locally. Pair by content, not path.
    args = contract["evaluation_env"]
    return {**contract, "evaluation_env": {k: v for k, v in args.items() if k != "npz_files"},
            "motion_sha256": list(contract["motion_sha256"].values())}


def original_tracking(qpos, original, geometry):
    import numpy as np
    from summarize_oldschool_policy_pair import DT, KNEES, OPENING, interval

    feet, width, _, _ = geometry.scan(qpos)
    orig_feet, orig_width, _, _ = geometry.scan(original)
    t = np.arange(len(qpos)) * DT
    knees = interval(t, *KNEES)
    opening = interval(t, *OPENING)
    knee_err = np.rad2deg(qpos[:, geometry.knees]) - np.rad2deg(original[:, geometry.knees])

    def rmse(delta, window):
        return float(np.sqrt(np.square(delta[window]).mean())) if window.any() else None

    foot = rmse(feet - orig_feet, knees)
    ankle = rmse(width - orig_width, opening)
    return {
        "both_foot_vertical_rmse_cm": 100 * foot if foot is not None else None,
        "left_knee_rmse_deg": rmse(knee_err[:, 0], knees),
        "right_knee_rmse_deg": rmse(knee_err[:, 1], knees),
        "ankle_width_rmse_cm": 100 * ankle if ankle is not None else None,
    }


def score_pair(runs, seed, digests, reference, original, target, duration_s):
    import numpy as np
    from summarize_oldschool_policy_pair import DT, JUMP, G1Geometry, interval, summarize

    model = runs["baseline"] / "g1_scene.mjb"
    model_sha = sha256(model)
    if sha256(runs["candidate"] / "g1_scene.mjb") != model_sha:
        raise ValueError(f"{reference}/seed{seed}: the two MuJoCo scenes differ")
    geometry = G1Geometry(model)
    orig_q, orig_joints = original
    goal_q, goal_joints = target
    np.testing.assert_array_equal(orig_joints, goal_joints)
    row, metadata = {"seed": seed}, {}
    for arm, run in runs.items():
        scores, meta = summarize(run, digests[arm], digests[reference], seed, geometry)
        _, _, frames, refq, joints, _ = meta
        np.testing.assert_array_equal(joints, goal_joints)
        if np.any((frames < 0) | (frames >= min(len(orig_q), len(goal_q)))):
            raise ValueError(f"{reference}/seed{seed}/{arm}: reference frames are out of range")
        if scores["completed"] and not math.isclose(scores["duration_s"], duration_s, abs_tol=1e-9):
            raise ValueError(f"{reference}/seed{seed}/{arm}: unexpected clip duration")
        # The compact motion library stores the goal quaternions and joints in float16.
        np.testing.assert_allclose(refq[:, 3:7], orig_q[frames, 3:7].astype(np.float16),
                                   atol=2e-6, rtol=0)
        np.testing.assert_allclose(refq[:, 7:], goal_q[frames, 7:].astype(np.float16),
                                   atol=2e-6, rtol=0)
        clip, = read_json(run / "report.json")["clips"]
        with np.load(run / Path(clip["states"]).name, allow_pickle=False) as states:
            q = states["qpos"].astype(float)
        outside = ~interval(np.arange(len(q)) * DT, 6., JUMP[1])
        landing = scores["landing"] or {}
        row[arm] = {
            **{k: scores[k] for k in (*FLAGS, "duration_s", "lower_foot_peak_cm",
                                     "whole_joint_rmse_deg", "arms_joint_rmse_deg",
                                     "outside_jump_joint_rmse_deg", "pelvis_position_mean_cm",
                                     "tail_mean_abs_yaw_deg", "terminal_abs_yaw_deg")},
            **original_tracking(q, orig_q[frames], geometry),
            "outside_jump_mean_pelvis_position_cm": (
                100 * float(np.linalg.norm(q[:, :3] - refq[:, :3], axis=1)[outside].mean())
                if outside.any() else None),
            "landing_tilt_deg": landing.get("max_pelvis_tilt_deg_next_300ms"),
            "contact_free_span_s": landing.get("contact_free_span_s"),
        }
        metadata[arm] = meta
    a, b = (metadata[arm] for arm in ARMS)
    n = min(len(a[2]), len(b[2]))
    for x, y in ((a[0], b[0]), (a[1], b[1]), (a[2][:n], b[2][:n]),
                 (a[3][:n], b[3][:n]), (a[4], b[4])):
        np.testing.assert_array_equal(x, y)
    if normalized_contract(a[5]) != normalized_contract(b[5]):
        raise ValueError(f"{reference}/seed{seed}: the two environment contracts differ")
    return row, (a[0], a[1], model_sha)


def run_comparison(paths, output, seeds, protocol):
    import numpy as np

    if not __debug__:
        raise RuntimeError("Run without Python optimization so recorder checks stay enabled")
    digests = input_digests(protocol)
    for key, path in paths.items():
        if sha256(path) != digests[key]:
            raise ValueError(f"{key}: input SHA-256 differs from {MANIFEST.name}: {path}")
    poses = {ref: pose_file(paths[ref]) for ref in REFERENCES}
    output.mkdir(parents=True)
    write_json(output / "plan.json", {"experiment": protocol["experiment"], "input_sha256": digests,
                                     "seeds": seeds})
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    results, starts = {}, {}
    for ref in REFERENCES:
        rows = []
        for seed in seeds:
            runs = {arm: output / ref / f"seed{seed}" / arm for arm in ARMS}
            for arm, run in runs.items():
                print(f"{ref} | start {seed} | {arm}", flush=True)
                record_one(paths[arm], paths[ref], seed, run)
            row, start = score_pair(runs, seed, digests, ref, poses["original"], poses[ref],
                                    protocol["duration_s"])
            if seed in starts:
                q0, v0, model_sha = starts[seed]
                np.testing.assert_array_equal(q0, start[0])
                np.testing.assert_array_equal(v0, start[1])
                if model_sha != start[2]:
                    raise ValueError(f"seed{seed}: the MuJoCo scenes differ between references")
            else:
                starts[seed] = start
            rows.append(row)
        results[ref] = rows
    if any(sha256(path) != digests[key] for key, path in paths.items()):
        raise ValueError("An input file changed during evaluation")
    data = comparison(protocol, results)
    validate_report(data, protocol)
    write_json(output / "comparison.json", data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="summarize saved per-start metrics using only Python")
    report.add_argument("--results", type=Path, default=RESULTS)
    report.add_argument("--include-display", action="store_true", help="include the demo start (0)")
    run = sub.add_parser("run", help="record and score the physical rollouts (CUDA environment)")
    for name in (*ARMS, *REFERENCES):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="a new output directory")
    run.add_argument("--seeds", nargs="+", type=int, help="subset of the locked starts; default: all")
    args = parser.parse_args()
    protocol = read_json(MANIFEST)
    if args.command == "report":
        print(markdown_report(read_json(args.results), protocol, include_display=args.include_display), end="")
        return
    seeds = seed_order(protocol) if args.seeds is None else args.seeds
    if not set(seeds) <= set(seed_order(protocol)) or len(seeds) != len(set(seeds)):
        run.error(f"choose distinct starts from {seed_order(protocol)}")
    if args.output.exists():
        run.error("--output must be a new directory")
    paths = {key: getattr(args, key).resolve() for key in protocol["inputs"]}
    data = run_comparison(paths, args.output.resolve(), seeds, protocol)
    print(markdown_report(data, protocol, include_display=(seeds == [protocol["display_seed"]])), end="")


if __name__ == "__main__":
    main()
