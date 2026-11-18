"""Build gravity-consistent low-hop (~13 cm) or higher-hop (~22 cm) Oldschool G1 references."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from summarize_oldschool_policy_pair import G1Geometry

FPS = 50
G = 9.81


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_motion(path):
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in f.files}
    pelvis = arrays["body_names"].tolist().index("pelvis")
    q = np.column_stack([arrays["body_pos_w"][:, pelvis], arrays["body_quat_w"][:, pelvis], arrays["joint_pos"]])
    return arrays, q.astype(np.float64)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10.0 + x * (-15.0 + 6.0 * x))


def export_reference(arrays, q, edit_window, source_label, out_path, overwrite=False):
    from supermimic.data.motion_convert import ResampledMotion, _so3_derivative, forward_kinematics, load_g1_model

    a_idx, b_idx = edit_window
    joint_vel = np.gradient(q[:, 7:], 1.0 / FPS, axis=0)
    rm = ResampledMotion(
        fps=FPS,
        root_pos=q[:, :3],
        root_quat=q[:, 3:7],
        joint_pos=q[:, 7:],
        root_lin_vel_w=np.gradient(q[:, :3], 1.0 / FPS, axis=0),
        root_ang_vel_w=_so3_derivative(q[:, 3:7], 1.0 / FPS),
        joint_vel=joint_vel,
    )
    body = forward_kinematics(load_g1_model(), rm)
    assert body.body_names == arrays["body_names"].tolist()
    full = {k: v.copy() for k, v in arrays.items()}
    updates = {
        "joint_pos": rm.joint_pos,
        "joint_vel": rm.joint_vel,
        "body_pos_w": body.pos_w,
        "body_quat_w": body.quat_w,
        "body_lin_vel_w": body.lin_vel_w,
        "body_ang_vel_w": body.ang_vel_w,
    }
    for key, val in updates.items():
        full[key][a_idx:b_idx + 1] = val[a_idx:b_idx + 1]
        assert np.isfinite(full[key]).all()
    pelvis = body.body_names.index("pelvis")
    full["body_quat_w"][:, pelvis] = arrays["body_quat_w"][:, pelvis]
    full["source"] = np.array(source_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing reference: {out_path}")
    with out_path.open("wb" if overwrite else "xb") as f:
        np.savez_compressed(f, **full)


def build_high_hop(raw_path, low_path, model_path, peak_m=0.22):
    arrays, raw = load_motion(raw_path)
    _, low = load_motion(low_path)
    assert raw.shape == low.shape == (597, 36)
    g = G1Geometry(model_path)
    m, d = g.model, g.data
    lo, hi = m.jnt_range[1:].T
    by_name = arrays["joint_names"].tolist()
    parts = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    leg = np.array([by_name.index(f"{s}_{p}_joint") for s in ("left", "right") for p in parts])
    sagittal = leg[[0, 3, 4, 6, 9, 10]]
    mass = m.body_mass.sum()
    a_idx, c_idx, j_idx, p_idx, l_idx, s_idx, b_idx = 300, 307, 315, 322, 330, 338, 348

    def geometry(row):
        d.qpos[:] = row
        mujoco.mj_kinematics(m, d)
        feet = (d.geom_xpos[g.feet, 2] - g.radius - g.half * np.abs(d.geom_xmat[g.feet, 8])).min(1) - d.geom_xpos[g.floor, 2]
        return m.body_mass @ d.xipos / mass, feet

    def ground(i, nominal, z):
        q = raw[i].copy()
        q[7 + leg] = nominal
        q[2] += 0.002 - geometry(q)[1].min()
        minimum = np.r_[0.5, lo[sagittal]] + 1e-7
        maximum = np.r_[1.0, hi[sagittal]] - 1e-7
        nominal_sag = q[7 + sagittal].copy()

        def resid(x):
            q[2], q[7 + sagittal] = x[0], x[1:]
            c, sole = geometry(q)
            return np.r_[(c[2] - z) / 0.004, (sole - 0.002) / 0.003, (x[1:] - nominal_sag) / 0.20]

        best = least_squares(resid, np.clip(np.r_[q[2], nominal_sag], minimum, maximum), bounds=(minimum, maximum), max_nfev=45, ftol=2e-7).x
        q[2], q[7 + sagittal] = best[0], best[1:]
        return q

    s_pose = ground(j_idx, low[316, 7 + leg], 0.731)
    e_pose = ground(l_idx, low[328, 7 + leg], 0.731)
    c0, _ = geometry(s_pose)
    c1, _ = geometry(e_pose)
    flight_t = (l_idx - j_idx) / FPS
    v0 = (c1[2] - c0[2]) / flight_t + G * flight_t / 2.0
    v1 = v0 - G * flight_t
    zpre = CubicHermiteSpline(np.array([a_idx, c_idx, j_idx]) / FPS, [geometry(low[a_idx])[0][2], 0.644, c0[2]], [0, 0, v0])
    zpost = CubicHermiteSpline(np.array([l_idx, s_idx, b_idx]) / FPS, [c1[2], 0.657, geometry(low[b_idx])[0][2]], [v1, 0, 0])
    pre = PchipInterpolator([a_idx, c_idx, j_idx], [raw[a_idx, 7 + leg], low[307, 7 + leg], s_pose[7 + leg]])
    post = PchipInterpolator([l_idx, s_idx, b_idx], [e_pose[7 + leg], low[337, 7 + leg], raw[b_idx, 7 + leg]])
    tuck = PchipInterpolator([j_idx, p_idx, l_idx], [s_pose[7 + leg], raw[p_idx, 7 + leg], e_pose[7 + leg]])
    spread = PchipInterpolator([j_idx, p_idx, l_idx], [s_pose[7 + leg], low[322, 7 + leg], e_pose[7 + leg]])

    def candidate(alpha):
        q = raw.copy()
        for i in range(a_idx, j_idx):
            q[i, 7 + leg] = pre(i)
            q[i, 2] += zpre(i / FPS) - geometry(q[i])[0][2]
        for i in range(l_idx + 1, b_idx + 1):
            q[i, 7 + leg] = post(i)
            q[i, 2] += zpost(i / FPS) - geometry(q[i])[0][2]
        for i in range(j_idx, l_idx + 1):
            t = (i - j_idx) / FPS
            q[i, 7 + leg] = (1.0 - alpha) * spread(i) + alpha * tuck(i)
            c = c0 + (c1 - c0) * t / flight_t
            c[2] = c0[2] + v0 * t - 0.5 * G * t * t
            q[i, :3] += c - geometry(q[i])[0]
        geom = [geometry(x) for x in q[j_idx:l_idx + 1]]
        return q, np.stack([x[0] for x in geom]), np.stack([x[1] for x in geom])

    strength = brentq(lambda a: candidate(a)[2].min(1).max() - peak_m, 0.0, 1.0)
    q, c, soles = candidate(strength)
    summary = {
        "mode": "high",
        "edit_window_s": [a_idx / FPS, b_idx / FPS],
        "planned_contact_boundaries_s": [j_idx / FPS, l_idx / FPS],
        "peak_lower_sole_cm": float(100.0 * soles.min(1).max()),
        "tuck_interpolation_fraction": float(strength),
        "source_sha256": digest(raw_path),
        "small_hop_sha256": digest(low_path),
    }
    return arrays, q, (a_idx, b_idx), summary


def build_low_hop(raw_path, model_path):
    arrays, qraw = load_motion(raw_path)
    q = qraw.copy()
    m = mujoco.MjModel.from_binary_path(str(model_path))
    d = mujoco.MjData(m)
    joints = arrays["joint_names"].tolist()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, k) or "" for k in range(m.ngeom)]
    fi = np.array([i for i, n in enumerate(names) if re.search(r"(?:left|right)_foot[1-7]_collision$", n)])
    rad = m.geom_size[fi, 0]
    half = np.where(m.geom_type[fi] == mujoco.mjtGeom.mjGEOM_CAPSULE, m.geom_size[fi, 1], 0.0)
    floor = next(i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE)
    mass = m.body_mass.sum()

    def geom(row):
        d.qpos[:] = row
        mujoco.mj_kinematics(m, d)
        z = d.geom_xpos[fi, 2] - rad - half * np.abs(d.geom_xmat[fi, 8]) - d.geom_xpos[floor, 2]
        return (m.body_mass @ d.xipos) / mass, np.array([z[:7].min(), z[7:].min()])

    idx = np.array([joints.index(f"{s}_{j}_joint") for s in ("left", "right") for j in ("hip_pitch", "knee", "ankle_pitch")])
    p = 7 + idx
    lo = np.r_[0.52, m.jnt_range[1 + idx, 0]]
    hi = np.r_[1.02, m.jnt_range[1 + idx, 1]]
    a_idx, c_idx, j_idx, l_idx, s_idx, b_idx = 300, 308, 316, 328, 336, 348

    def grounded(i, nominal, h):
        row = qraw[i].copy()
        row[p] = nominal
        row[2] += 0.002 - geom(row)[1].min()
        init = np.clip(np.r_[row[2], nominal], lo + 1e-6, hi - 1e-6)

        def residual(v):
            row[2], row[p] = v[0], v[1:]
            c, sole = geom(row)
            return np.r_[(c[2] - h) / 0.004, (sole - 0.002) / 0.003, (v[1:] - nominal) / 0.23]

        v = least_squares(residual, init, bounds=(lo, hi), max_nfev=30, ftol=2e-7).x
        row[2], row[p] = v[0], v[1:]
        return row.copy()

    start = grounded(j_idx, np.array([-0.11, 0.27, 0.11] * 2), 0.735)
    end = grounded(l_idx, np.array([-0.17, 0.35, -0.15] * 2), 0.735)
    a0, a1 = geom(start)[0], geom(end)[0]
    flight_t = (l_idx - j_idx) / FPS
    v0 = (a1[2] - a0[2]) / flight_t + G * flight_t / 2.0
    v1 = v0 - G * flight_t
    zfloor = lambda i: geom(qraw[i])[0][2] - geom(qraw[i])[1].min() + 0.002
    pre_nom = PchipInterpolator([a_idx, c_idx, j_idx], np.stack([qraw[a_idx, p], np.array([-0.50, 1.01, -0.49] * 2), start[p]]), axis=0)
    post_nom = PchipInterpolator([l_idx, s_idx, b_idx], np.stack([end[p], np.array([-0.33, 0.69, -0.33] * 2), qraw[b_idx, p]]), axis=0)
    zpre = CubicHermiteSpline(np.array([a_idx, c_idx, j_idx]) / FPS, [zfloor(a_idx), 0.640, a0[2]], [0, 0, v0])
    zpost = CubicHermiteSpline(np.array([l_idx, s_idx, b_idx]) / FPS, [a1[2], 0.660, zfloor(b_idx)], [v1, 0, 0])
    flight_nom = PchipInterpolator([j_idx, (j_idx + l_idx) // 2, l_idx], np.stack([start[p], np.array([-0.68, 1.38, -0.42] * 2), end[p]]), axis=0)
    for i in range(a_idx, b_idx + 1):
        if i < j_idx:
            q[i, p] = pre_nom(i)
            q[i, 2] += zpre(i / FPS) - geom(q[i])[0][2]
        elif i <= l_idx:
            t = (i - j_idx) / FPS
            q[i, p] = flight_nom(i)
            c = a0 + (a1 - a0) * t / flight_t
            c[2] = a0[2] + v0 * t - 0.5 * G * t * t
            q[i, :3] += c - geom(q[i])[0]
        else:
            q[i, p] = post_nom(i)
            q[i, 2] += zpost(i / FPS) - geom(q[i])[0][2]
    soles = np.array([geom(q[i])[1].min() for i in range(a_idx, b_idx + 1)])
    summary = {
        "mode": "low",
        "edit_window_s": [a_idx / FPS, b_idx / FPS],
        "planned_contact_boundaries_s": [j_idx / FPS, l_idx / FPS],
        "peak_lower_sole_cm": float(100.0 * soles.max()),
        "source_sha256": digest(raw_path),
    }
    return arrays, q, (a_idx, b_idx), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("low", "high"), required=True)
    parser.add_argument("--raw", type=Path, default=ROOT / "outputs/motions/oldschool_full_v1.npz")
    parser.add_argument("--low-hop", type=Path, help="Existing low-hop NPZ required for --mode high")
    parser.add_argument("--model", type=Path, required=True, help="Compiled MuJoCo g1_scene.mjb")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-cm", type=float, default=22.0)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "low":
        arrays, q, win, summary = build_low_hop(args.raw, args.model)
        label = f"{args.raw} + local ballistic low hop"
    else:
        if args.low_hop is None:
            parser.error("--low-hop is required for --mode high")
        arrays, q, win, summary = build_high_hop(args.raw, args.low_hop, args.model, peak_m=args.peak_cm / 100.0)
        label = f"{args.raw} + shared higher hop"

    if args.write:
        export_reference(arrays, q, win, label, args.output, overwrite=args.overwrite)
        summary["output"] = str(args.output)
        summary["motion_sha256"] = digest(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
