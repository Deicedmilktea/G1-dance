"""Capture closed-loop teacher demonstrations and fine-tune actor decoder layers via supervised imitation."""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jump_teacher import ASYM, COMMON, JOINTS, KNOTS, LegTargets, contact_gate, sha256


def capture(args):
    from supermimic.policy.nets import Policy, PolicyCfg
    from supermimic.runner import make_env

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = Policy(PolicyCfg(**ckpt["config"]["policy"])).to("cuda:0").eval()
    policy.load_state_dict(ckpt["policy"], strict=True)

    teacher_x = np.load(args.teacher_params) if args.teacher_params else np.concatenate([COMMON.ravel(), ASYM.ravel()])
    paired = args.num_pairs
    env = make_env(
        motion_dir=str(args.motion.parent),
        clip_name=args.motion.stem,
        num_envs=2 * paired,
        device="cuda:0",
        obs_noise=False,
        reset_noise=True,
        seed=args.seed,
    )
    try:
        term = env.action_manager.get_term("joint_pos")
        scale, offset = term._scale.clone(), term._offset.clone()
        leg = LegTargets(env, np.broadcast_to(teacher_x, (2 * paired, len(teacher_x))).copy(), record=False)
        # Zero out compensation for the first `paired` anchor environments; apply teacher offsets to second half.
        leg.profiles[:paired] = 0.0
        cmd = env.command_manager.get_term("motion")
        cmd.forced_frame[:] = 0
        obs, _ = env.reset()
        d = env.sim.mj_data
        m = env.sim.mj_model
        # Mirror perturbed initial poses from teacher half to baseline half? Keep identical per pair:
        # Both halves receive same paired states by resetting with shared initial conditions.
        init_q, init_v = d.qpos.clone(), d.qvel.clone()
        h = ckpt["config"]["policy"]["history"]
        action_history = h * 29
        act_hist = obs["policy"][:, -action_history:].reshape(2 * paired, h, 29).clone()
        length = cmd.library.clips[0].length
        traces = {
            k: torch.empty((length, 2 * paired, width), device=env.device)
            for k, width in (("obs_student", 930), ("goal", 680), ("action_teacher", 29), ("action_raw", 29))
        }
        q = torch.empty((length + 1, 2 * paired, 36), device=env.device)
        q[0] = d.qpos
        end = torch.zeros(2 * paired, dtype=torch.long, device=env.device)
        done = torch.zeros(2 * paired, dtype=torch.bool, device=env.device)
        for i in range(length):
            ob, goal = obs["policy"], obs["goal"]
            aligned = torch.cat((ob[:, :-action_history], act_hist.flatten(1)), 1)
            a = policy(ob, goal)
            traces["obs_student"][i] = aligned
            traces["goal"][i] = goal
            traces["action_raw"][i] = a
            obs, _, fail, timeout, _ = env.step(a)
            motor = term._processed_actions
            diff = motor - (a * scale + offset)
            teacher = a + diff / scale
            traces["action_teacher"][i] = teacher
            q[i + 1] = d.qpos
            act_hist = torch.cat((act_hist[:, 1:], teacher[:, None]), dim=1)
            term_now = fail | timeout
            first = term_now & (end == 0)
            end[first] = i + 1
            done[first] = (~fail & env.termination_manager.get_term("clip_end"))[first]
            if (end != 0).all():
                break
        arrays = {k: v.cpu().numpy() for k, v in dict(qpos=q, initial_qpos=init_q, initial_qvel=init_v, end=end, completed=done, scale=scale[0], **traces).items()}
        mujoco.mj_saveModel(m, str(out / "g1_scene.mjb"), None)
        eligible = []
        for k in range(paired):
            j = paired + k
            gate = contact_gate(m, arrays["qpos"][: arrays["end"][j] + 1, j], bool(arrays["completed"][j]))
            if gate["passes"]:
                eligible.append(k)
        np.savez(out / "supervision.npz", **arrays, joint_names=np.array(leg.names), eligible_pairs=np.array(eligible))
        (out / "capture_report.json").write_text(json.dumps({"paired_starts": paired, "eligible_pairs": eligible}, indent=2) + "\n")
        print(f"Captured {len(eligible)}/{paired} qualified teacher demonstrations in {out}")
    finally:
        env.close()


def train(args):
    from supermimic.policy.nets import Policy, PolicyCfg

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    original = ck["policy"]
    actor = Policy(PolicyCfg(**cfg["policy"])).to("cuda:0").eval()
    actor.load_state_dict(original, strict=True)
    actor.requires_grad_(False)
    linear = [x for x in actor.decoder.modules() if isinstance(x, nn.Linear)]
    assert len(linear) == 5 and linear[-1].out_features == 29
    for layer in linear[-3:]:
        layer.requires_grad_(True)
    trainable = [(k, x) for k, x in actor.named_parameters() if x.requires_grad]
    optim = torch.optim.Adam([x for _, x in trainable], lr=args.lr)

    with np.load(args.data, allow_pickle=False) as f:
        s = {k: f[k] for k in ("obs_student", "goal", "action_teacher", "end", "eligible_pairs", "joint_names", "scale")}
    t, n, _ = s["obs_student"].shape
    paired = n // 2
    tt = np.arange(t)
    leg = torch.as_tensor([x in JOINTS for x in s["joint_names"]], device="cuda:0")
    scale = torch.as_tensor(s["scale"], device="cuda:0")
    train_ids = list(range(1, max(2, int(paired * 0.75))))
    val_ids = list(range(max(2, int(paired * 0.75)), paired))

    def build_split(ids, teach):
        ids_arr = np.asarray(ids)
        time = (tt >= KNOTS[0]) & (tt <= KNOTS[-1])
        lane = np.intersect1d(ids_arr, s["eligible_pairs"]) + paired if teach else ids_arr
        if not teach:
            time = ~time
        mask = (tt[:, None] < s["end"][None]) & time[:, None] & np.isin(np.arange(n), lane)[None]
        return [torch.from_numpy(s[k][mask]).to("cuda:0") for k in ("obs_student", "goal", "action_teacher")]

    td, ad = build_split(train_ids, True), build_split(train_ids, False)
    tv, av = build_split(val_ids, True), build_split(val_ids, False)

    def errors(ds):
        return (actor(ds[0], ds[1]) - ds[2]) * scale

    @torch.no_grad()
    def evaluate(step):
        def collect(ds):
            return torch.cat([errors([x[i:i + 512] for x in ds]) for i in range(0, len(ds[0]), 512)])
        j, b = collect(tv), collect(av)
        l_err = j[:, leg].square().mean()
        o_err = j[:, ~leg].square().mean()
        a_err = b.square().mean()
        obj = 0.8 * l_err + 0.2 * o_err + args.anchor_weight * a_err
        return {
            "step": step,
            "teach_legs_rmse_deg": math.degrees(math.sqrt(float(l_err))),
            "teach_other_rmse_deg": math.degrees(math.sqrt(float(o_err))),
            "outside_anchor_rmse_deg": math.degrees(math.sqrt(float(a_err))),
            "objective": float(obj),
        }

    def pick(ds, size):
        idx = torch.randint(len(ds[0]), (size,), device="cuda:0")
        return [x[idx] for x in ds]

    log = [evaluate(0)]
    best, chosen = None, None
    for step in range(1, args.steps + 1):
        dt, da = pick(td, 192), pick(ad, 320)
        jt = errors(dt)
        loss_teach = 0.8 * jt[:, leg].square().mean() + 0.2 * jt[:, ~leg].square().mean()
        loss_anchor = errors(da).square().mean()
        loss = loss_teach + args.anchor_weight * loss_anchor
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x for _, x in trainable], 1.0, error_if_nonfinite=True)
        optim.step()
        if step % args.eval_interval == 0 or step == args.steps:
            row = evaluate(step)
            log.append(row)
            print(json.dumps(row), flush=True)
            if best is None or row["objective"] < best["objective"]:
                best = row
                chosen = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}

    assert chosen is not None
    dest = out / "ckpt_actor.pt"
    torch.save({"policy": chosen, "config": copy.deepcopy(cfg), "iteration": int(ck["iteration"])}, dest)
    (out / "training_report.json").write_text(json.dumps({
        "selected": best,
        "validation": log,
        "checkpoint_sha256": sha256(dest),
        "starting_checkpoint_sha256": sha256(args.checkpoint),
    }, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--motion", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--teacher-params", type=Path)
    c.add_argument("--num-pairs", type=int, default=33)
    c.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train")
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--steps", type=int, default=1200)
    t.add_argument("--eval-interval", type=int, default=300)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--anchor-weight", type=float, default=3.0)
    t.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    capture(args) if args.action == "capture" else train(args)


if __name__ == "__main__":
    main()
