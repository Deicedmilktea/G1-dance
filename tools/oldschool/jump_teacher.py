"""Leg-target actuator offset teacher & jump kinematic evaluator (EXP-010/011/013/017)."""

import hashlib
import math
import re

import mujoco
import numpy as np
import torch

JOINTS = tuple(
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
)
KNOTS = (300, 307, 315, 323, 331, 339, 348)  # 6.00 .. 6.96 s at 50 Hz
DT = 0.02
STEPS = np.arange(295, 386)
FLIGHT = (310, 332)
KNEES = (316, 328)
OPENING = (318, 332)
TRACK = (314, 334)

COMMON = np.array([
    [-0.14, 0.02, 0.30, -0.18, -0.02],
    [0.02, 0.03, -0.18, 0.22, -0.03],
    [-0.25, 0.14, 0.65, -0.25, -0.09],
    [-0.08, 0.07, 0.22, -0.10, -0.05],
    [0.02, 0.02, -0.04, 0.03, -0.02],
])
ASYM = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.04, 0.0, 0.0, -0.03],
    [0.0, 0.02, 0.0, 0.0, -0.01],
    [0.0, 0.0, 0.0, 0.0, 0.0],
])


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def curve(x):
    """Interpolate 5 interior knot leg offsets (left/right, 5 joints) over KNOTS."""
    c, a = x[:25].reshape(5, 5), x[25:].reshape(5, 5)
    inner = np.concatenate([c + a, c - a], axis=-1)
    pad = np.zeros((*inner.shape[:-2], 1, len(JOINTS)))
    values = np.concatenate([pad, inner, pad], axis=-2)
    steps = np.arange(KNOTS[0], KNOTS[-1] + 1)
    index = np.searchsorted(KNOTS[1:], steps, side="right").clip(0, len(KNOTS) - 2)
    phase = (steps - KNOTS[index]) / (KNOTS[index + 1] - KNOTS[index])
    w = (phase ** 2 * (3 - 2 * phase))[:, None]
    return ((1 - w) * values[..., index, :] + w * values[..., index + 1, :]).astype("float32")


def foot_geometry(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    ids = np.array([
        [i for i, n in enumerate(names) if re.search(fr"/?{side}_foot[1-7]_collision$", n)]
        for side in ("left", "right")
    ])
    assert ids.shape == (2, 7)
    ids = ids.ravel()
    radii = model.geom_size[ids, 0]
    half = np.where(model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_CAPSULE, model.geom_size[ids, 1], 0.0)
    floor, = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    return ids, radii, half, int(floor)


def model_indices(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    ankles = np.array([
        next(i for i, n in enumerate(names) if n.rsplit("/", 1)[-1] == f"{s}_ankle_roll_link")
        for s in ("left", "right")
    ])
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)]
    knees = np.array([
        model.jnt_qposadr[next(i for i, n in enumerate(names) if n.rsplit("/", 1)[-1] == f"{s}_knee_joint")]
        for s in ("left", "right")
    ])
    return ankles, knees


def yaw_np(q):
    q = q[..., 3:7]
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(q):
    q = q[..., 3:7]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(1.0 - 2.0 * np.square(q[..., 1:3]).sum(-1), -1.0, 1.0)))


def read_kinematics(model, q):
    data = mujoco.MjData(model)
    ids, rad, half, floor = foot_geometry(model)
    ankles, knees = model_indices(model)
    xyzw, feet = [], []
    for state, psi in zip(q, yaw_np(q)):
        data.qpos[:] = state
        mujoco.mj_kinematics(model, data)
        sole = (
            data.geom_xpos[ids, 2] - rad - half * np.abs(data.geom_xmat[ids, 8]) - data.geom_xpos[floor, 2]
        ).reshape(2, 7).min(1)
        dx = data.xpos[ankles] - state[:3]
        c, s = math.cos(psi), math.sin(psi)
        xyzw.append(dx @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
        feet.append(sole)
    return np.asarray(xyzw), np.asarray(feet), np.asarray(q[:, knees])


def contact_gate(model, states, completed):
    """Verify airborne phase, stable post-landing support, and no non-foot contact."""
    data = mujoco.MjData(model)
    ids, rad, half, floor = foot_geometry(model)
    foot_set = set(ids.tolist())
    soles = np.empty((len(states), 2))
    nonfoot = 0
    for i, row in enumerate(states):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        soles[i] = (
            data.geom_xpos[ids, 2] - rad - half * np.abs(data.geom_xmat[ids, 8]) - data.geom_xpos[floor, 2]
        ).reshape(2, 7).min(1)
        for c in data.contact[:data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if floor in pair and not (pair & foot_set):
                nonfoot += 1
    win = np.arange(round(6.12 / DT), round(7.12 / DT) + 1)
    low = soles[:, :].min(1)
    air = low[win] > 0.02
    best = 0
    cur = 0
    end = -1
    for i, hit in enumerate(air):
        cur = cur + 1 if hit else 0
        if cur > best:
            best, end = cur, int(win[i])
    stable = False
    if best >= 2 and end + 15 < len(states):
        post = slice(end + 1, end + 16)
        stable = bool(
            (low[post] <= 0.01).sum() >= 9
            and tilt_deg(states[post]).max() <= 25.0
            and states[post, 2].min() >= 0.55
        )
    passes = bool(completed and nonfoot == 0 and best >= 2 and stable)
    return {
        "completed": bool(completed),
        "nonfoot_contacts": int(nonfoot),
        "airborne_steps": int(best),
        "lower_foot_peak_cm": float(low[win].max() * 100.0),
        "stable_landing": stable,
        "passes": passes,
    }


class LegTargets:
    """Add smooth knot offsets at the actuator target stage; preserve actor action history."""

    def __init__(self, env, x, record=False):
        term = env.action_manager.get_term("joint_pos")
        names = tuple(n.rsplit("/", 1)[-1] for n in term.target_names)
        assert len(names) == 29 and set(JOINTS) <= set(names)
        model = env.sim.mj_model
        limits = {}
        for i in model.actuator_trnid[:, 0]:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)).rsplit("/", 1)[-1]
            limits[name] = model.jnt_range[i].copy()
        lo, hi = np.stack([limits[j] for j in JOINTS]).T
        self.idx = torch.tensor([names.index(j) for j in JOINTS], device=env.device)
        self.lo = torch.tensor(lo, device=env.device, dtype=torch.float32)
        self.hi = torch.tensor(hi, device=env.device, dtype=torch.float32)
        self.names = names
        self.device = env.device
        self.record = record
        self.deltas = []
        self.step = 0
        self.set_profiles(x)
        original = term.process_actions

        def process(actions):
            original(actions)
            n = self.step
            self.step += 1
            if not (KNOTS[0] < n < KNOTS[-1]):
                if self.record:
                    self.deltas.append(torch.zeros_like(term._processed_actions))
                return
            target = term._processed_actions
            before = target[:, self.idx]
            offset = self.profiles[:, n - KNOTS[0]]
            after = torch.clamp(before + offset, torch.minimum(before, self.lo), torch.maximum(before, self.hi))
            target[:, self.idx] = after
            if self.record:
                change = torch.zeros_like(target)
                change[:, self.idx] = after - before
                self.deltas.append(change)

        term.process_actions = process

    def set_profiles(self, x):
        self.profiles = torch.as_tensor(curve(x), device=self.device)
        self.step = 0
