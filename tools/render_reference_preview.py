"""Label a source/reference preview produced by the upstream NJU renderer."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oldschool/pipeline.json")
    parser.add_argument("--front-frame", type=int, help="A source frame facing the camera, used only to select preview view direction")
    args = parser.parse_args()
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    settings = json.loads(args.config.read_text())
    source, csv = ROOT / settings["video"], ROOT / settings["reference_csv"]
    rows = np.loadtxt(csv, delimiter=",", ndmin=2)
    capture = cv2.VideoCapture(str(source))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rows) == settings["frames"]
    fps = capture.get(cv2.CAP_PROP_FPS)
    assert fps == settings["source_fps"]
    capture.release()
    raw, final = csv.parent / "reference_preview_raw.mp4", csv.parent / "reference_preview.mp4"
    assert not raw.exists() and not final.exists()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(ROOT / ".runtime/bin/ffmpeg")
    upstream = ROOT / "third_party/SuperMimic-Repro-Sim2Real/tools/video2motion/render_preview.py"
    command = [sys.executable, str(upstream), str(csv), str(source), str(raw),
               "--robot_xml", str(ROOT / settings["robot_xml"]), "--height", "720"]
    camera_yaw = None
    if args.front_frame is not None:
        assert 0 <= args.front_frame < len(rows)
        q = rows[max(0, args.front_frame-3):args.front_frame+4, 3:7]
        x, y, z, w = q.T
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        camera_yaw = float(np.degrees(np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean())))
        command.extend(["--cam_yaw", str(camera_yaw)])
    (csv.parent / "render_command.json").write_text(json.dumps({"command": command,
        "upstream_renderer_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
        "camera_front_frame": args.front_frame, "camera_yaw_deg": camera_yaw,
        "scope": "View direction only; source and robot poses are unchanged."}, indent=2)+"\n")
    subprocess.run(command, cwd=ROOT, check=True)
    capture = cv2.VideoCapture(str(raw))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rows)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_width = width-int(720*4/3)
    samples = set(range(0, len(rows), 30)) | {len(rows)-1}
    diagnostic = json.loads((csv.parent / "reference_check.json").read_text())
    samples.update(row["to_frame"] for row in diagnostic["fastest_joint_transitions"][:4])
    try:
        with imageio.get_writer(str(final), fps=fps, codec="libx264", macro_block_size=1,
                                ffmpeg_params=["-crf", "18", "-movflags", "+faststart"]) as writer:
            for i in range(len(rows)):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Rendered preview ended early")
                frame[:54] = 0
                cv2.putText(frame, "SOURCE VIDEO", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "G1 REFERENCE MOTION | POSE PLAYBACK", (source_width+12, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
                for x0 in [12, source_width+12]:
                    cv2.putText(frame, f"t={i/fps:.2f}s | frame {i}/{len(rows)-1}", (x0, 46), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1, cv2.LINE_AA)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if i in samples:
                    cv2.imwrite(str(csv.parent / f"reference_preview_frame{i:03d}.jpg"), frame)
    finally:
        capture.release()
    subprocess.run([str(ROOT / ".runtime/bin/ffmpeg"), "-v", "error", "-i", str(final), "-f", "null", "-"], check=True)
    result = {"frames": len(rows), "fps": fps, "width": width, "height": height,
              "duration_s": len(rows)/fps, "complete_decode": True, "last_source_frame_included": True,
              "output": final.relative_to(ROOT).as_posix(), "sampled_frames": sorted(samples),
              "scope": "Source frames paired with reference poses; kinematic playback, no learned-policy execution."}
    (csv.parent / "preview_check.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
