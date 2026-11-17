"""Overlay cached person boxes and optional COCO joints on a source video."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
         (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pipeline = json.loads((ROOT / "configs/oldschool/pipeline.json").read_text())
    parser.add_argument("--video", type=Path, default=ROOT / pipeline["video"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    args.run = (args.run or ROOT / "outputs/gvhmr" / args.video.stem).resolve()
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch

    boxes = torch.load(args.run / "preprocess/bbx.pt", map_location="cpu", weights_only=True)["bbx_xyxy"].numpy()
    keypoint_path = args.run / "preprocess/vitpose.pt"
    points = torch.load(keypoint_path, map_location="cpu", weights_only=True).numpy() if keypoint_path.exists() else None
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Expected finite person boxes with shape (frames, 4).")
    if points is not None and (points.shape != (len(boxes), 17, 3) or not np.isfinite(points).all()):
        raise ValueError("Expected finite COCO joints with shape (frames, 17, 3).")
    diagnostics = json.loads((args.run / "preprocess/tracking_diagnostics.json").read_text())
    detected = set(diagnostics["detected_frames"])
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(boxes):
        capture.release()
        raise ValueError("Input video and cached boxes must have identical frame counts.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    output = args.run / ("joints_check.mp4" if points is not None else "tracking_check.mp4")
    samples = set(range(0, len(boxes), max(1, round(fps)))) | {len(boxes) - 1}
    samples.update(set(range(len(boxes))) - detected)
    if points is not None:
        samples.update(np.argsort(points[:, :, 2].min(axis=1))[:8].tolist())
    try:
        with imageio.get_writer(str(output), fps=fps, codec="libx264", macro_block_size=1,
                                ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"]) as writer:
            for index, box in enumerate(boxes):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Video decoding ended before the cached sequence.")
                color = (0, 220, 0) if index in detected else (0, 165, 255)
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"Frame {index + 1}/{len(boxes)} | {index / fps:.2f}s"
                if index not in detected:
                    label += " | BOX FILLED"
                cv2.putText(frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                if points is not None:
                    joints = points[index]
                    valid = joints[:, 2] >= args.confidence
                    for a, b in EDGES:
                        if valid[a] and valid[b]:
                            cv2.line(frame, tuple(joints[a, :2].astype(int)), tuple(joints[b, :2].astype(int)), (255, 220, 0), 3)
                    for joint in joints[valid]:
                        cv2.circle(frame, tuple(joint[:2].astype(int)), 4, (0, 255, 255), -1)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if index in samples:
                    sample = output.parent / (output.stem + f"_frame{index:03d}.jpg")
                    cv2.imwrite(str(sample), frame)
    finally:
        capture.release()
    report = {
        "frames": len(boxes), "fps": fps, "selected_track_id": diagnostics["selected_track_id"],
        "selected_track_detected_frames": len(detected),
        "filled_box_frames_zero_based": sorted(set(range(len(boxes))) - detected),
        "with_2d_joints": points is not None, "output": str(output.relative_to(ROOT)),
        "scope": "Person boxes and optional two-dimensional joints; no three-dimensional or physical validation.",
    }
    (output.parent / (output.stem + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
