"""Phase-aligned G1 jump rewards for staged higher-hop RL (EXP-017) and direct NJU RL (EXP-018)."""
from __future__ import annotations

import math
import re

import mujoco
import numpy as np
import torch

from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse, yaw_quat

DT = 0.02

STAGED_PHASES = {
    "push": (6.12, 6.30, 0.04),
    "air": (6.30, 6.62, 0.04),
    "shape": (6.32, 6.60, 0.04),
    "landing": (6.56, 6.78, 0.06),
}
STAGED_WEIGHTS = {"push": 1.4, "com": 1.1, "soles": 2.6, "knees": 1.2, "span": 1.0, "landing": 1.3}
STAGED_SCALES = {
    "push_m_s": 0.35, "com_m": 0.045, "sole_m": 0.050, "knee_rad": 0.30,
    "span_m": 0.075, "landing_sole_m": 0.050, "landing_com_v_m_s": 0.45,
}
STAGED_ANCHOR_OUTSIDE_S = (5.80, 7.02)
STAGED_ANCHOR_STRENGTH = 0.16

DIRECT_PHASES = {
    "push": (6.14, 6.32, 0.04),
    "air": (6.23, 6.61, 0.06),
    "shape": (6.29, 6.69, 0.05),
    "landing": (6.57, 6.98, 0.09),
    "heading": (9.80, 12.01, 0.16),
}
DIRECT_WEIGHTS = {"push": 2.0, "soles": 3.5, "knees": 1.2, "span": 1.15, "landing": 1.15, "heading": 0.45}
DIRECT_CLEARANCE_CAP_M = 0.22
DIRECT_PUSH_CAP_M_S = 1.48
DIRECT_ANCHOR_OUTSIDE_S = (5.80, 7.03)
DIRECT_ANCHOR_STRENGTH = 0.18


def ramp(t, start, end, edge):
    up = ((t - (start - edge)) / edge).clamp(0.0, 1.0)
    down = (((end + edge) - t) / edge).clamp(0.0, 1.0)
    return 0.5 * (1.0 - torch.cos(math.pi * torch.minimum(up, down)))


def bounded_loss(error, scale):
    ratio = (error / scale).square()
    return ratio / (1.0 + ratio)


class HopGeometry:
    """Precomputed MuJoCo foot/COM/knee geometry for vectorized G1 reward evaluation."""

    def __init__(self, model, device, dtype=torch.float32):
        self.fk = G1Kinematics(model, device=device, dtype=dtype)
        arr = lambda x: torch.as_tensor(np.array(x), device=device, dtype=dtype)
        self.mass = arr(model.body_mass[1:])
        self.mass = self.mass / self.mass.sum()
        self.local_com = arr(model.body_ipos[1:])
        geoms = [
            i for side in ("left", "right")
            for i in range(model.ngeom)
            if re.fullmatch(f"{side}_foot[1-7]_collision", model.geom(i).name)
        ]
        assert len(geoms) == 14
        capsule, sphere = int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_SPHERE)
        assert all(int(model.geom_type[g]) in (capsule, sphere) for g in geoms)
        self.geom_body = torch.as_tensor(model.geom_bodyid[geoms] - 1, device=device, dtype=torch.long)
        self.geom_pos = arr(model.geom_pos[geoms])
        z = arr(np.broadcast_to([0.0, 0.0, 1.0], (14, 3)))
        self.geom_dir = quat_apply(arr(model.geom_quat[geoms]), z)
        self.geom_radius = arr(model.geom_size[geoms, 0])
        self.geom_half = arr(np.where(model.geom_type[geoms] == capsule, model.geom_size[geoms, 1], 0.0))
        self.ankles = self.fk.body_indices(("left_ankle_roll_link", "right_ankle_roll_link"))
        self.knee = torch.as_tensor(
            [model.joint(x + "_knee_joint").qposadr[0] for x in ("left", "right")],
            device=device,
            dtype=torch.long,
        )

    def features(self, state):
        qpos, qvel = state[..., :36], state[..., 36:]
        info = self.fk(qpos, qvel)
        q = info["quat"]
        offs = quat_apply(q, self.local_com.expand_as(info["pos"]))
        com_z = ((info["pos"] + offs)[..., 2] * self.mass).sum(-1)
        com_vz = ((info["lin_vel"] + torch.cross(info["ang_vel"], offs, dim=-1))[..., 2] * self.mass).sum(-1)
        foot_q = q[..., self.geom_body, :]
        foot_pos = info["pos"][..., self.geom_body, :] + quat_apply(foot_q, self.geom_pos.expand_as(foot_q[..., :3]))
        foot_axis = quat_apply(foot_q, self.geom_dir.expand_as(foot_q[..., :3]))
        sole = foot_pos[..., 2] - self.geom_radius - self.geom_half * foot_axis[..., 2].abs()
        soles = sole.reshape(*state.shape[:-1], 2, 7).amin(-1)
        delta = info["pos"][..., self.ankles[0], :] - info["pos"][..., self.ankles[1], :]
        lateral = quat_apply_inverse(yaw_quat(qpos[..., 3:7]), delta)[..., 1].abs()
        return torch.cat((com_z[..., None], com_vz[..., None], soles, qpos[..., self.knee], lateral[..., None]), -1)


class HopReward:
    """Staged higher-hop phase-aligned cost (EXP-017)."""

    def __init__(self, geometry, reference_states):
        self.geometry = geometry
        with torch.no_grad():
            self.ref = geometry.features(reference_states)
        self.clip_last = self.ref.shape[0] - 1

    def costs(self, s, frame):
        want = self.ref[frame]
        actual = self.geometry.features(s)
        dz, dv, zl, zr, kl, kr, width = (actual - want).unbind(-1)
        t = frame.to(actual.dtype) * DT
        phase = {k: ramp(t, *v) for k, v in STAGED_PHASES.items()}
        zerr = torch.stack((zl, zr), -1)
        kerr = torch.stack((kl, kr), -1)
        landing = (
            0.65 * bounded_loss(zerr, STAGED_SCALES["landing_sole_m"]).mean(-1)
            + 0.35 * bounded_loss(dv, STAGED_SCALES["landing_com_v_m_s"])
        )
        return {
            "push": phase["push"] * bounded_loss(dv, STAGED_SCALES["push_m_s"]),
            "com": phase["air"] * bounded_loss(dz, STAGED_SCALES["com_m"]),
            "soles": phase["air"] * bounded_loss(zerr, STAGED_SCALES["sole_m"]).mean(-1),
            "knees": phase["shape"] * bounded_loss(kerr, STAGED_SCALES["knee_rad"]).mean(-1),
            "span": phase["shape"] * bounded_loss(width, STAGED_SCALES["span_m"]),
            "landing": phase["landing"] * landing,
        }

    def __call__(self, state, frame):
        if state.shape[:-1] != frame.shape:
            raise ValueError("Reward state/frame shapes disagree")
        active = (frame > round(STAGED_PHASES["push"][0] / DT)) & (frame < round(STAGED_PHASES["landing"][1] / DT))
        frame, state = frame.reshape(-1), state.reshape(-1, state.shape[-1])
        ids = active.reshape(-1).nonzero(as_tuple=True)[0]
        out = torch.zeros_like(frame, dtype=state.dtype)
        if ids.numel():
            terms = self.costs(state[ids], frame[ids])
            out = out.index_copy(0, ids, -sum(STAGED_WEIGHTS[k] * v for k, v in terms.items()))
        return out.reshape(active.shape)


class DirectReward:
    """Training-only phase cost on raw GVHMR/G1 states for direct NJU RL (EXP-018)."""

    def __init__(self, geometry, reference_states):
        self.geometry = geometry
        with torch.no_grad():
            self.ref = geometry.features(reference_states)
            self.ref_heading = yaw_quat(reference_states[:, 3:7])
        self.clip_last = self.ref.shape[0] - 1

    def costs(self, s, frame):
        want = self.ref[frame]
        actual = self.geometry.features(s)
        _, vz, zl, zr, kl, kr, width = actual.unbind(-1)
        _, want_vz, wl, wr, wkl, wkr, ww = want.unbind(-1)
        t = frame.to(actual.dtype) * DT
        phase = {k: ramp(t, *v) for k, v in DIRECT_PHASES.items()}
        actual_sole = torch.stack((zl, zr), -1)
        desired_sole = torch.stack((wl, wr), -1).clamp(0.0, DIRECT_CLEARANCE_CAP_M)
        deficit = (desired_sole - actual_sole).clamp_min(0.0)
        per_foot = torch.nn.functional.smooth_l1_loss(
            deficit, torch.zeros_like(deficit), beta=0.035, reduction="none"
        ).clamp_max(0.31) / 0.22
        clearance = 0.60 * per_foot.mean(-1) + 0.40 * per_foot.amax(-1)
        near_floor = torch.minimum(zl, zr)
        landing = (
            0.60 * bounded_loss((near_floor - 0.035).clamp_min(0.0), 0.065)
            + 0.25 * bounded_loss((-vz - 0.85).clamp_min(0.0), 0.40)
            + 0.15 * bounded_loss((vz - 0.30).clamp_min(0.0), 0.35)
        )
        knee_err = torch.stack((kl - wkl, kr - wkr), -1)
        heading_dot = (yaw_quat(s[:, 3:7]) * self.ref_heading[frame]).sum(-1)
        heading = (1.0 - heading_dot.square()).clamp_min(0.0) / (math.sin(0.20) ** 2)
        heading = heading / (1.0 + heading)
        return {
            "push": phase["push"] * bounded_loss(
                (want_vz.clamp_min(0.0).mul(1.18).clamp_max(DIRECT_PUSH_CAP_M_S) - vz).clamp_min(0.0), 0.35
            ),
            "soles": phase["air"] * clearance,
            "knees": phase["shape"] * bounded_loss(knee_err, 0.42).mean(-1),
            "span": phase["shape"] * bounded_loss(width - ww, 0.075),
            "landing": phase["landing"] * landing,
            "heading": phase["heading"] * heading,
        }

    def __call__(self, state, frame):
        if state.shape[:-1] != frame.shape:
            raise ValueError("Reward state/frame shapes disagree")
        tt = frame * DT
        use = (tt > DIRECT_PHASES["push"][0]) & (tt < DIRECT_PHASES["landing"][1])
        tail = ramp(tt, *DIRECT_PHASES["heading"]).to(state.dtype)
        frame, state = frame.reshape(-1), state.reshape(-1, state.shape[-1])
        heading_dot = (yaw_quat(state[:, 3:7]) * self.ref_heading[frame]).sum(-1)
        heading = (1.0 - heading_dot.square()).clamp_min(0.0) / (math.sin(0.20) ** 2)
        out = -DIRECT_WEIGHTS["heading"] * tail.reshape(-1) * heading / (1.0 + heading)
        ids = use.reshape(-1).nonzero(as_tuple=True)[0]
        if ids.numel():
            terms = self.costs(state[ids], frame[ids])
            out = out.index_add(0, ids, -sum(DIRECT_WEIGHTS[k] * v for k, v in terms.items() if k != "heading"))
        return out.reshape(use.shape)
