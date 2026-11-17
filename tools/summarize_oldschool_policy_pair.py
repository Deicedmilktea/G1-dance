"""Summarize the Oldschool policy comparison from matched physical trajectories."""

import argparse
import json
from pathlib import Path
import re

import mujoco
import numpy as np

from record_policy_pair import ARMS, sha256, write_json

JUMP = (6.12, 7.12)
KNEES = (6.32, 6.56)
OPENING = (6.36, 6.64)
TAIL = 10.
DT = .02
PHYSICS_DT = .005
SIDES = ("left", "right")


def interval(t, lo, hi):
    return (t >= lo - 1e-9) & (t <= hi + 1e-9)


def blocks(mask):
    indices = np.flatnonzero(mask)
    return [x for x in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(x)]


def yaw(q):
    rotation = q[:, 3:7]
    w, x, y, z = (rotation / np.linalg.norm(rotation, axis=1, keepdims=True)).T
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def tilt(q):
    rotation = q[:, 3:7]
    rotation = rotation / np.linalg.norm(rotation, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(1 - 2*np.square(rotation[:, 1:3]).sum(1), -1, 1)))


class G1Geometry:
    def __init__(self, path):
        m = mujoco.MjModel.from_binary_path(str(path))
        self.model = m
        self.data = mujoco.MjData(m)
        assert (m.nq, m.nv, m.nu) == (36, 35, 29)
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or str(i)
                      for i in range(m.ngeom)]
        self.feet = np.array([[i for i, name in enumerate(self.names)
                               if re.search(rf"(?:^|/){side}_foot[1-7]_collision$", name)]
                              for side in SIDES])
        assert self.feet.shape == (2, 7)
        self.left, self.right = map(set, self.feet)
        self.floor, = np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        capsule = m.geom_type[self.feet] == mujoco.mjtGeom.mjGEOM_CAPSULE
        assert np.all(capsule | (m.geom_type[self.feet] == mujoco.mjtGeom.mjGEOM_SPHERE))
        self.radius = m.geom_size[self.feet, 0]
        self.half = np.where(capsule, m.geom_size[self.feet, 1], 0.)
        bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i).rsplit("/", 1)[-1]: i
                  for i in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)}
        joints = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i).rsplit("/", 1)[-1]: i
                  for i in range(m.njnt) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)}
        self.ankles = [bodies[f"{s}_ankle_roll_link"] for s in SIDES]
        self.knees = [int(m.jnt_qposadr[joints[f"{s}_knee_joint"]]) for s in SIDES]

    def scan(self, q, *, contacts=False):
        d, m = self.data, self.model
        lower, width, foot_contacts, other = [], [], [], set()
        for state, psi in zip(q, yaw(q)):
            d.qpos[:] = state
            if contacts:
                mujoco.mj_forward(m, d)
            else:
                mujoco.mj_kinematics(m, d)
            feet = self.feet
            soles = d.geom_xpos[feet, 2] - self.radius - self.half * np.abs(d.geom_xmat[feet, 8])
            lower.append(soles.min(axis=1) - d.geom_xpos[self.floor, 2])
            dx, dy = (d.xpos[self.ankles[0]] - d.xpos[self.ankles[1]])[:2]
            width.append(-np.sin(psi) * dx + np.cos(psi) * dy)
            if contacts:
                touched = set()
                for c in d.contact[:d.ncon]:
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if c.dist <= 0 and self.floor in (g1, g2):
                        touched.add(g2 if g1 == self.floor else g1)
                foot_contacts.append((bool(touched & self.left), bool(touched & self.right)))
                other.update(self.names[g] for g in touched - self.left - self.right)
        return np.asarray(lower), np.asarray(width), np.asarray(foot_contacts), sorted(other)


def summarize(run, checkpoint_sha, motion_sha, seed, geo):
    report = json.loads((run / "report.json").read_text())
    contract = json.loads((run / "environment_contract.json").read_text())
    assert report["checkpoint_sha256"] == checkpoint_sha and report["seed"] == seed
    assert report["reset_noise"] == bool(seed) and report["resets_during_each_recorded_clip"] == 0
    assert not report["weight_updates"] and not report["external_forces"]
    assert list(contract["motion_sha256"].values()) == [motion_sha]
    assert not contract["auto_reset"] and contract["physics_dt_s"] == PHYSICS_DT
    assert contract["control_dt_s"] == DT and contract["gravity"] == [0, 0, -9.81]
    clip, = report["clips"]
    with np.load(run / Path(clip["states"]).name, allow_pickle=False) as data:
        q, ref, frame, fps, names = [data[k] for k in ("qpos", "qref", "frame", "fps", "joint_names")]
    with np.load(run / "physics_trace.npz", allow_pickle=False) as data:
        q0, v0, physics, force = [data[k] for k in
                                 ("initial_qpos", "initial_qvel", "physics_qpos", "external_force_max")]
    assert fps.tolist() == [1 / DT] and len(q) == clip["steps"] + 1
    assert frame[0] == 1 and np.isin(np.diff(frame), [0, 1]).all()
    assert len(physics) == len(force) == clip["steps"] * 4 and not force.any()
    assert np.isfinite(q).all() and np.isfinite(ref).all() and np.isfinite(physics).all()
    np.testing.assert_array_equal(q0, q[0])
    np.testing.assert_allclose(physics[3::4], q[1:], atol=1e-7, rtol=0)
    t, tp = np.arange(len(q)) * DT, np.arange(1, len(physics) + 1) * PHYSICS_DT
    feet, width, _, _ = geo.scan(q)
    ref_feet, ref_width, _, _ = geo.scan(ref)
    physics_feet, _, touches, other = geo.scan(physics, contacts=True)
    lower, ref_lower = feet.min(1), ref_feet.min(1)
    flight = max(blocks(interval(t, *JUMP) & (lower > .02)), key=len, default=[])
    finished, landing = bool(clip["completed"]), None
    if len(flight) >= 2:
        mid = t[flight[len(flight) // 2]]
        free = next((b for b in blocks((physics_feet.min(1) > 0) & ~touches.any(1))
                     if tp[b[0]] - 1e-9 <= mid <= tp[b[-1]] + 1e-9), None)
        if free is not None:
            hit = np.flatnonzero((tp > tp[free[-1]] + 1e-9) & touches.any(1))
            if len(hit):
                first = int(hit[0])
                time = float(tp[first])
                after = interval(tp, time, time + .30)
                support = int((interval(t, time, time + .30) & (lower <= .01)).sum())
                lean, height = float(tilt(physics[after]).max()), float(physics[after, 2].min())
                landing = {"first_touch_s": time, "contact_free_span_s": float(tp[free[-1]] - tp[free[0]]),
                           "first_foot": "both" if touches[first].all() else SIDES[int(touches[first, 1])],
                           "max_pelvis_tilt_deg_next_300ms": lean,
                           "min_pelvis_height_m_next_300ms": height, "support_samples": support,
                           "stable": bool(tp[-1] + 1e-9 >= time + .30 and support >= 9 and
                                          lean <= 25 and height >= .55 and not other)}
    no_fall = finished and not other
    dy = yaw(q.astype(float)) - yaw(ref.astype(float))
    err = np.rad2deg(np.arctan2(np.sin(dy), np.cos(dy)))
    j, k, w = interval(t, *JUMP), interval(t, *KNEES), interval(t, *OPENING)
    qerr = np.rad2deg(q[:, 7:].astype(float) - ref[:, 7:].astype(float))
    root_err = 100 * np.linalg.norm(q[:, :3].astype(float) - ref[:, :3].astype(float), axis=1)
    knee = np.rad2deg(q[:, geo.knees].astype(float) - ref[:, geo.knees].astype(float))
    arms = [i for i, name in enumerate(names) if any(x in name for x in ("shoulder", "elbow", "wrist"))]
    assert arms
    outside = ~interval(t, 6., JUMP[1])
    near_reference = np.flatnonzero(j & (t > t[np.argmax(np.where(j, ref_lower, -np.inf))]) &
                                    (ref_lower <= .01)) if j.any() else []
    scores = {
        "duration_s": clip["simulated_seconds"], "completed": finished,
        "termination_flags": clip["termination_flags"], "nonfoot_ground_contacts": other,
        "completed_without_fall": no_fall,
        "sustained_jump": len(flight) >= 2,
        "jump_and_stable_landing": bool(landing and landing["stable"] and no_fall),
        "landing": landing,
        "lower_foot_peak_cm": 100*float(lower[j].max()) if j.any() else None,
        "reference_lower_foot_peak_cm": 100*float(ref_lower[j].max()) if j.any() else None,
        "reference_first_foot_near_ground_s": float(t[near_reference[0]]) if len(near_reference) else None,
        "knee_rmse_deg": float(np.sqrt(np.square(knee[k]).mean())) if k.any() else None,
        "right_knee_rmse_deg": float(np.sqrt(np.square(knee[k, 1]).mean())) if k.any() else None,
        "ankle_spacing_rmse_cm": 100*float(np.sqrt(np.square(width[w] - ref_width[w]).mean())) if w.any() else None,
        "whole_joint_rmse_deg": float(np.sqrt(np.square(qerr).mean())),
        "arms_joint_rmse_deg": float(np.sqrt(np.square(qerr[:, arms]).mean())),
        "outside_jump_joint_rmse_deg": float(np.sqrt(np.square(qerr[outside]).mean())),
        "pelvis_position_mean_cm": float(root_err.mean()),
        "whole_mean_abs_yaw_deg": float(np.abs(err).mean()),
        "tail_mean_abs_yaw_deg": float(np.abs(err[t >= TAIL]).mean()) if finished and t[-1] >= TAIL else None,
        "terminal_abs_yaw_deg": float(abs(err[-1])) if finished else None,
    }
    contract.pop("checkpoint_env")
    return scores, (q0, v0, frame, ref, names, contract)


def aggregate(rows):
    paired_complete = [r for r in rows if all(r[k]["completed"] for k in ARMS)]
    counts = ("completed", "completed_without_fall", "sustained_jump", "jump_and_stable_landing")
    errors = ("lower_foot_peak_cm", "whole_joint_rmse_deg", "arms_joint_rmse_deg",
              "outside_jump_joint_rmse_deg", "pelvis_position_mean_cm", "whole_mean_abs_yaw_deg",
              "tail_mean_abs_yaw_deg", "terminal_abs_yaw_deg", "knee_rmse_deg", "right_knee_rmse_deg",
              "ankle_spacing_rmse_cm")
    return {
        "paired_completions": len(paired_complete),
        **{k: {"counts": {n: sum(r[k][n] for r in rows) for n in counts},
               "means_among_paired_completions": {
                   n: float(np.mean([r[k][n] for r in paired_complete])) if paired_complete and
                   all(r[k][n] is not None for r in paired_complete) else None for n in errors}}
           for k in ARMS},
        "candidate_error_improved_counts": {
            n: sum(r["candidate"][n] < r["baseline"][n] for r in paired_complete
                   if r["candidate"][n] is not None and r["baseline"][n] is not None)
            for n in errors if n != "lower_foot_peak_cm"},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", type=Path)
    args = p.parse_args()
    folder = args.run_root
    plan = json.loads((folder / "plan.json").read_text())
    assert not plan["weight_updates"] and not plan["action_compensation"]
    assert json.loads((folder / "finished.json").read_text())["input_hashes_unchanged"]
    out = {"plan": plan, "matched_initial_pose_and_velocity": True,
           "matched_reference_frames_within_each_motion": True, "matched_evaluation_contract": True,
           "metrics": {"jump_window_s": JUMP, "knee_window_s": KNEES, "opening_window_s": OPENING,
                       "tail_from_s": TAIL, "ground": "non-positive contact distance in saved 200 Hz states"},
           "results": {}}
    starts = {}
    for motion, digest in plan["motion_sha256"].items():
        rows = []
        for seed in plan["seeds"]:
            group = folder / motion / f"seed{seed}"
            model_path = group / "baseline/g1_scene.mjb"
            model_sha = sha256(model_path)
            assert sha256(group / "candidate/g1_scene.mjb") == model_sha
            row, meta = {"seed": seed, "model_sha256": model_sha}, {}
            geo = G1Geometry(model_path)
            for arm in ARMS:
                row[arm], meta[arm] = summarize(group / arm, plan["policy_sha256"][arm], digest, seed, geo)
            a, b = meta["baseline"], meta["candidate"]
            n = min(len(a[2]), len(b[2]))
            for x, y in ((a[0], b[0]), (a[1], b[1]), (a[2][:n], b[2][:n]),
                         (a[3][:n], b[3][:n]), (a[4], b[4])):
                np.testing.assert_array_equal(x, y)
            assert a[5] == b[5], "Policy comparison changed the evaluation environment"
            if seed in starts:
                q0, v0 = starts[seed]
                np.testing.assert_array_equal(a[0], q0)
                np.testing.assert_array_equal(a[1], v0)
            else:
                starts[seed] = a[:2]
            rows.append(row)
        out["results"][motion] = {"aggregate": aggregate(rows), "rows": rows}
        print(motion, json.dumps(out["results"][motion]["aggregate"], indent=2), flush=True)
    write_json(folder / "comparison.json", out)


if __name__ == "__main__":
    main()
