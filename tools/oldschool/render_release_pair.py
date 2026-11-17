"""Compose a side-by-side video pairing a source dance clip with a recorded MuJoCo rollout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FFMPEG = ROOT / ".runtime/bin/ffmpeg"
BG, FG, MUTED, ACCENT, DIVIDER = "#121820", "#f3f4ef", "#b0bdca", "#f2c86e", "#344250"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def render_pair(args):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if args.software_egl:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    with np.load(args.states, allow_pickle=False) as z:
        actual = z["qpos"].copy()
        reference = z["qref"].copy()
        frames = z["frame"].copy()

    if args.studio:
        canvas, pane, origins = (1152, 1024), (576, 1024), ((0, 0), (576, 0))
    else:
        canvas, pane, origins = (1360, 1360), (648, 1134), ((24, 174), (688, 174))

    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), pane[0])
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), pane[1])
    renderer = mujoco.Renderer(model, height=pane[1], width=pane[0])
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)

    video = imageio.get_reader(str(args.video))
    n_source = video.count_frames()
    sim_ids = np.round(np.linspace(0, len(actual) - 1, args.num_output_frames)).astype(int)
    source_ids = [min(n_source - 1, round(int(frames[i]) * args.source_fps / args.sim_fps)) for i in sim_ids]

    base = Image.new("RGB", canvas, BG)
    mask = Image.new("L", pane, 0)
    radius = 0 if args.studio else 10
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pane[0] - 1, pane[1] - 1), radius=radius, fill=255)
    if not args.studio:
        d = ImageDraw.Draw(base)
        title_font = ImageFont.truetype(subprocess.check_output(["fc-match", "-f", "%{file}", "DejaVu Sans:style=Bold"], text=True), 62)
        d.text((24, 32), "Oldschool", font=title_font, fill=FG)
        d.line((24, 115, 1336, 115), fill=DIVIDER, width=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = str(args.ffmpeg if args.ffmpeg.is_file() else "ffmpeg")
    try:
        with tempfile.TemporaryDirectory(dir=args.output.parent, prefix="encode-") as td:
            staged = Path(td) / "pair.mp4"
            cmd = [
                ffmpeg_bin, "-nostdin", "-v", "error", "-xerror",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{canvas[0]}x{canvas[1]}",
                "-framerate", str(args.output_fps), "-i", "pipe:0",
                "-i", str(args.video), "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-movflags", "+faststart", "-y", str(staged),
            ]
            with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
                assert enc.stdin
                try:
                    for sim_idx, src_idx in zip(sim_ids, source_ids):
                        src_img = Image.fromarray(video.get_data(src_idx)).resize(pane, Image.Resampling.LANCZOS)
                        camera.lookat[:] = (*reference[sim_idx, :2], args.lookat_z)
                        data.qpos[:] = actual[sim_idx]
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera, scene_option=options)
                        sim_img = Image.fromarray(renderer.render().copy())
                        canvas_img = base.copy()
                        canvas_img.paste(src_img, origins[0], mask)
                        canvas_img.paste(sim_img, origins[1], mask)
                        enc.stdin.write(canvas_img.tobytes())
                finally:
                    enc.stdin.close()
                if enc.wait():
                    raise subprocess.CalledProcessError(enc.returncode, cmd)
            staged.replace(args.output)
        print(f"Rendered {args.output}")
    finally:
        video.close()
        renderer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--states", type=Path, required=True, help="Recorded *_states.npz from record_policy_pair/physics")
    p.add_argument("--model", type=Path, required=True, help="Compiled MuJoCo g1_scene.mjb")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    p.add_argument("--studio", action="store_true", help="Clean full-bleed dual pane without header bar")
    p.add_argument("--software-egl", action="store_true")
    p.add_argument("--distance", type=float, default=3.0)
    p.add_argument("--azimuth", type=float, default=90.0)
    p.add_argument("--elevation", type=float, default=-10.0)
    p.add_argument("--lookat-z", type=float, default=0.82)
    p.add_argument("--source-fps", type=float, default=30.0)
    p.add_argument("--sim-fps", type=float, default=50.0)
    p.add_argument("--output-fps", type=int, default=25)
    p.add_argument("--num-output-frames", type=int, default=300)
    render_pair(p.parse_args())


if __name__ == "__main__":
    main()
