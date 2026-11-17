"""Render recorded reference and physical policy states with one shared camera."""

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy-label", default="PRETRAINED POLICY | PHYSICS")
    parser.add_argument("--baseline-run", type=Path, help="Add a previously recorded baseline physics panel")
    parser.add_argument("--baseline-label", default="PRETRAINED POLICY | PHYSICS")
    parser.add_argument("--show-heading-error", action="store_true", help="Label each panel's signed pelvis heading error")
    parser.add_argument("--camera-distance", type=float, default=2.9, help="Shared reference-following view distance; increase to retain large global drift")
    args = parser.parse_args()
    if not math.isfinite(args.camera_distance) or args.camera_distance <= 0:
        parser.error("camera-distance must be positive and finite")
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text())
    if len(report["clips"]) != 1:
        parser.error("Render one recorded clip per run.")
    output = run / "reference_vs_policy.mp4"
    if output.exists():
        raise FileExistsError(output)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(ROOT / ".runtime/bin/ffmpeg")
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    clip = report["clips"][0]
    with np.load(clip["states"], allow_pickle=False) as states:
        actual = states["qpos"].copy()
        reference = states["qref"].copy()
        metrics = states["metrics"].copy()
        reference_frames = states["frame"].copy()
        fps = float(states["fps"][0])
    assert actual.shape == reference.shape and actual.shape[1] == 36
    assert len(actual) == clip["steps"] + 1
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    assert report["resets_during_each_recorded_clip"] == 0
    baseline = baseline_metrics = None
    if args.baseline_run is not None:
        baseline_report = json.loads((args.baseline_run / "report.json").read_text())
        assert baseline_report["seed"] == report["seed"]
        assert baseline_report["reset_noise"] == report["reset_noise"]
        assert baseline_report["resets_during_each_recorded_clip"] == 0
        with np.load(baseline_report["clips"][0]["states"], allow_pickle=False) as states:
            baseline = states["qpos"][:len(actual)].copy()
            baseline_metrics = states["metrics"][:len(actual)].copy()
            assert baseline.shape == actual.shape
            assert np.array_equal(states["qref"][:len(actual)], reference)
            assert np.array_equal(states["frame"][:len(actual)], reference_frames)
    model = mujoco.MjModel.from_binary_path(str(run / "g1_scene.mjb"))
    assert model.nq == 36 and model.nu == 29
    width, height = 640, 480
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.quality.shadowsize = 2048
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    w, x, y, z = reference[0, 3:7]
    heading = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    camera.distance, camera.azimuth, camera.elevation = args.camera_distance, heading + 180, -12
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    indices = list(range(0, len(actual), 2))
    if indices[-1] != len(actual) - 1:
        indices.append(len(actual) - 1)
    selected = {0, len(indices) // 2, len(indices) - 1}
    try:
        with imageio.get_writer(str(output), fps=fps / 2, codec="libx264", macro_block_size=1,
                                ffmpeg_params=["-crf", "18", "-movflags", "+faststart"]) as writer:
            for ordinal, index in enumerate(indices):
                # Both panels follow the reference root, so global drift stays visible.
                camera.lookat[:] = [reference[index, 0], reference[index, 1], 0.72]
                panels = []
                panel_poses = [(reference[index], "REFERENCE MOTION")]
                if baseline is not None:
                    panel_poses.append((baseline[index], args.baseline_label))
                panel_poses.append((actual[index], args.policy_label))
                for pose, title in panel_poses:
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    panel = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, width, 80 if args.show_heading_error else 56), fill=(18, 24, 32))
                    draw.text((10, 6), title, font=font, fill="white")
                    draw.text((10, 30), f"t={index / fps:.2f}s | reference frame {reference_frames[index]} | seed {report['seed']}", font=font, fill="white")
                    if args.show_heading_error:
                        def yaw(p):
                            w, x, y, z = p[3:7] / np.linalg.norm(p[3:7])
                            return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                        delta = yaw(pose)-yaw(reference[index])
                        heading_error = math.degrees(math.atan2(math.sin(delta), math.cos(delta)))
                        draw.text((10, 54), f"Heading error {heading_error:+.1f} deg", font=font, fill="white")
                    panels.append(np.asarray(panel))
                combined = Image.fromarray(np.concatenate(panels, axis=1))
                draw = ImageDraw.Draw(combined)
                draw.rectangle((0, height - 34, width * len(panel_poses), height), fill=(18, 24, 32))
                message = (f"Root error {metrics[index, 0]:.3f} m | joint RMSE {metrics[index, 3]:.3f} rad"
                           f" | tilt error {metrics[index, 2]:.1f} deg | no mid-clip reset")
                if baseline_metrics is not None:
                    baseline_name = args.baseline_label.split("|", 1)[0].strip()
                    policy_name = args.policy_label.split("|", 1)[0].strip()
                    message = (f"{baseline_name}: root {baseline_metrics[index, 0]:.3f} m / joints {baseline_metrics[index, 3]:.3f} rad"
                               f" | {policy_name}: root {metrics[index, 0]:.3f} m / joints {metrics[index, 3]:.3f} rad"
                               " | both physical rollouts, no mid-clip reset")
                draw.text((10, height - 28), message, font=font, fill="white")
                writer.append_data(np.asarray(combined))
                if ordinal in selected:
                    combined.save(run / f"physics_comparison_state{index:03d}.jpg")
    finally:
        renderer.close()
    result = {
        "output": str(output), "video_frames": len(indices), "video_fps": fps / 2,
        "captured_states": len(actual), "physical_duration_s": clip["simulated_seconds"],
        "recorded_state_indices": indices, "terminal_state_included": indices[-1] == len(actual) - 1,
        "completed": clip["completed"], "failed": clip["failed"],
        "policy_label": args.policy_label,
        "baseline_label": args.baseline_label if baseline is not None else None,
        "heading_error_labels": args.show_heading_error,
        "camera_distance_m": args.camera_distance,
        "baseline_run": str(args.baseline_run.resolve()) if args.baseline_run is not None else None,
        "timing_note": "Replay includes the initial and first terminal state. Display sampling is every two control ticks, with the terminal state always retained.",
        "scope": "Offline rendering of previously recorded physical states; all panels use the reference-centered camera.",
    }
    (run / "render_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "recorded_state_indices"}, indent=2))


if __name__ == "__main__":
    main()
