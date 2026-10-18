"""Visual checks: annotated video, timeline/trajectory plot, contact sheets."""
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

from . import config as C
from .video import open_video, read_frame, iter_frames

# BGR colours by how the label was obtained
COLORS = {"tag_hit": (0, 200, 0), "same_track": (130, 220, 130), "track_unsupported": (200, 200, 60), "track_before_tag": (200, 100, 255),
          "fill": (0, 140, 255), "interpolated": (0, 255, 255)}
GREY = (170, 170, 170)


def _load(video):
    out = C.out_dir(Path(video))
    for name in ("person_detections.csv", "staff_positions.csv"):
        if not (out / name).exists():
            raise FileNotFoundError(f"{out / name} missing. Run `people` and `link` first.")
    det = pd.read_csv(out / "person_detections.csv")
    staff = pd.read_csv(out / "staff_positions.csv")
    hp = out / "tag_hits.csv"
    hits = pd.read_csv(hp) if hp.exists() else pd.DataFrame(columns=["frame", "tag_x", "tag_y", "conf"])
    by = lambda d: {int(f): g for f, g in d.groupby("frame")}
    return out, det, staff, hits, by(det), by(staff), by(hits)


def _draw(frame, f, det_f, staff_f, hits_f):
    """Grey boxes + track ids for everyone, coloured box for staff, red circles for tag hits."""
    if det_f is not None:
        for r in det_f.itertuples():
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), GREY, 1)
            cv2.putText(frame, f"id{int(r.track_id)}", (int(r.x1), max(12, int(r.y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
    lines = []                                   # one status line per staff identity
    if staff_f is not None:
        for r in staff_f.itertuples():
            col = COLORS.get(r.source, (255, 255, 255))
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 3)
            cv2.putText(frame, f"STAFF {int(r.staff_id)}", (int(r.x1), int(r.y2) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            # the reported Task 2 coordinate: the tag centre when a tag was seen, else the box centre
            x, y = int(round(r.x)), int(round(r.y))
            cv2.drawMarker(frame, (x, y), col, cv2.MARKER_CROSS, 16, 2)
            lines.append((f"STAFF {int(r.staff_id)}  x={x} y={y}  ({r.source})", col))
    if not lines:
        lines.append(("no staff", (170, 170, 170)))
    if hits_f is not None:
        for h in hits_f.itertuples():
            cv2.circle(frame, (int(h.tag_x), int(h.tag_y)), 10, (0, 0, 255), 2)

    cv2.putText(frame, f"frame {f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for i, (text, col) in enumerate(lines):
        cv2.putText(frame, text, (10, 62 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    return frame


def render(video, trail_len=60):
    """Annotated video: python run.py render --video ..."""
    out, _, _, _, det_by, staff_by, hits_by = _load(video)
    cap = open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    path = out / "staff_annotated.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trails = {}
    for f, frame in iter_frames(video):
        s = staff_by.get(f)
        if s is not None:
            for r in s.itertuples():
                trails.setdefault(int(r.staff_id), deque(maxlen=trail_len)).append((int(r.cx), int(r.cy)))
        for trail in trails.values():
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (255, 200, 0), 2)
        writer.write(_draw(frame, f, det_by.get(f), s, hits_by.get(f)))
    writer.release()
    print("Saved", path)
    return path


def timeline(video):
    """Coverage by source over time + path on a background frame -> timeline.png"""
    out, _, staff, hits, *_ = _load(video)
    cap = open_video(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = read_frame(cap, total // 2)
    cap.release()

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 2]})
    rgb = lambda c: tuple(v / 255 for v in c[::-1])
    for i, (src, col) in enumerate(COLORS.items()):
        fr = staff[staff.source == src].frame
        ax[0].vlines(fr, i, i + 0.8, color=rgb(col), label=f"{src} ({len(fr)})")
    if len(hits):
        ax[0].vlines(hits.frame, len(COLORS), len(COLORS) + 0.8, color="red", label="tag hit")
    ax[0].set_xlim(0, total)
    ax[0].set_yticks([])
    ax[0].set_xlabel("frame")
    ax[0].set_title(f"Staff coverage: {staff.frame.nunique()} of {total} frames, {staff.staff_id.nunique()} identity(ies)")
    ax[0].legend(loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.25))

    if bg is not None:
        ax[1].imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    for src, col in COLORS.items():
        s = staff[staff.source == src]
        ax[1].scatter(s.cx, s.cy, s=6, color=rgb(col), label=src)
    ax[1].set_title("Staff path (box centre) by label source")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    path = out / "timeline.png"
    plt.savefig(path, dpi=150)
    print("Saved", path)
    return path


def sheet(video, start, end, step, cols=3, tile_w=480, max_tiles=24):
    """Contact sheet of annotated frames: python run.py sheet --video ... --start 0 --end 363 --step 20"""
    out, _, _, _, det_by, staff_by, hits_by = _load(video)
    cap = open_video(video)
    frames = list(range(start, end + 1, step))[:max_tiles]
    tiles = []
    for f in frames:
        fr = read_frame(cap, f)
        if fr is None:
            continue
        fr = _draw(fr, f, det_by.get(f), staff_by.get(f), hits_by.get(f))
        h, w = fr.shape[:2]
        tiles.append(cv2.resize(fr, (tile_w, int(tile_w * h / w))))
    cap.release()
    if not tiles:
        raise RuntimeError("No frames read for that range.")
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    path = out / f"sheet_{start}_{end}.jpg"
    cv2.imwrite(str(path), img)
    print("Saved", path, f"({len(frames)} frames)")
    return path