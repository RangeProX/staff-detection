"""Stage 1: detect and track people -> outputs/<video>/person_detections.csv"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from . import config
from .video import open_video, read_frame

COLUMNS = ["frame", "timestamp", "track_id", "x1", "y1", "x2", "y2", "confidence"]


def run(video):
    video = Path(video)
    out = config.out_dir(video)
    previews = out / "previews"
    previews.mkdir(exist_ok=True)

    model = YOLO(config.PERSON_WEIGHTS)      # fresh model = fresh tracker state per video
    cap = open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps:.2f}  Total frames: {total}")

    every = int(fps * config.PREVIEW_EVERY_SEC) if fps > 0 and config.PREVIEW_EVERY_SEC else 0
    rows, f = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame, persist=True, tracker=config.TRACKER, classes=[0],
            conf=config.PERSON_CONF, imgsz=config.PERSON_IMGSZ, verbose=False,
        )[0]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            rows.append({
                "frame": f,
                "timestamp": f / fps if fps else 0.0,
                "track_id": int(box.id[0]) if box.id is not None else -1,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": float(box.conf[0]),
            })

        if every and f % every == 0:
            cv2.imwrite(str(previews / f"detected_{f}.jpg"), result.plot())
        f += 1
    cap.release()

    df = pd.DataFrame(rows, columns=COLUMNS)
    path = out / "person_detections.csv"
    df.to_csv(path, index=False)
    print(f"Processed {f} frames, {len(df)} person detections -> {path}")
    return path


def probe(video, weights=None, imgsz=None, conf=None, start=0, end=300, step=20,
          cols=3, tile_w=480, max_tiles=24):
    """Try a person-detector setting on a few frames (no tracking) and save a contact sheet.
    python run.py probe --video ... --weights yolo11m.pt --imgsz 1280 --conf 0.1 --start 0 --end 363 --step 20"""
    weights = weights or config.PERSON_WEIGHTS
    imgsz = imgsz or config.PERSON_IMGSZ
    conf = conf or 0.1
    model = YOLO(weights)
    cap = open_video(video)
    tiles, counts = [], []
    for f in list(range(start, end + 1, step))[:max_tiles]:
        frame = read_frame(cap, f)
        if frame is None:
            continue
        r = model.predict(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        vis = r.plot()
        cv2.putText(vis, f"f{f}  people={len(r.boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        h, w = vis.shape[:2]
        tiles.append(cv2.resize(vis, (tile_w, int(tile_w * h / w))))
        counts.append(len(r.boxes))
    cap.release()
    if not tiles:
        raise RuntimeError("No frames read for that range.")
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    path = config.out_dir(video) / f"probe_{Path(weights).stem}_{imgsz}_{start}_{end}.jpg"
    cv2.imwrite(str(path), img)
    print(f"Saved {path}; detections per sampled frame: {counts}")
    return path


# ---- zero-label detector sweep -------------------------------------------------------------
# Each entry: (name, weights or None = config default, imgsz or None, TTA, 4-way rotation)
SWEEP = [
    ("current settings",   None,          None, False, False),
    ("+ TTA",              None,          None, True,  False),
    ("imgsz 1280",         None,          1280, False, False),
    ("imgsz 1280 + TTA",   None,          1280, True,  False),
    ("4-way rotation",     None,          None, False, True),
    ("yolo11m @1280",      "yolo11m.pt",  1280, False, False),
]

_ROT = [
    (None, lambda x, y, W, H: (x, y)),
    (cv2.ROTATE_90_CLOCKWISE, lambda x, y, W, H: (H - 1 - y, x)),
    (cv2.ROTATE_180, lambda x, y, W, H: (W - 1 - x, H - 1 - y)),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda x, y, W, H: (y, W - 1 - x)),
]


def _covers(boxes, x, y, pad):
    """Does any detected box contain (x, y), allowing a margin of `pad` box sizes?"""
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        m = pad * max(x2 - x1, y2 - y1)
        if x1 - m <= x <= x2 + m and y1 - m <= y <= y2 + m:
            return True
    return False


def sweep(video, conf=None, pad=0.1, frames=None, max_frames=24):
    """Which zero-label setting finds the staff in the frames the current detector misses?

    Uses ground_truth.csv (your clicks) and person_detections.csv to pick those frames, then tries
    each setting in SWEEP and reports how many it finds. No new labelling.
    python run.py sweep --video videos/sample.mp4
    """
    out = config.out_dir(video)
    gt_path, det_path = out / "ground_truth.csv", out / "person_detections.csv"
    for p in (gt_path, det_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run `annotate` and `people` first.")
    conf = conf or config.PERSON_CONF

    gt = pd.read_csv(gt_path)
    gt = gt[gt.visible == 1]
    det = pd.read_csv(det_path)
    det_by = {int(f): g for f, g in det.groupby("frame")}

    if frames:
        todo = [(int(f), *gt[gt.frame == int(f)][["x", "y"]].iloc[0]) for f in frames]
    else:
        todo = []
        for r in gt.itertuples():                      # keep only the detector's misses
            g = det_by.get(int(r.frame))
            hit = False
            if g is not None:
                m = pad * np.maximum(g.x2 - g.x1, g.y2 - g.y1)
                hit = bool(((g.x1 - m <= r.x) & (r.x <= g.x2 + m)
                            & (g.y1 - m <= r.y) & (r.y <= g.y2 + m)).any())
            if not hit:
                todo.append((int(r.frame), float(r.x), float(r.y)))
    todo = todo[:max_frames]
    if not todo:
        print("No missed frames: the current detector already has a box on the staff everywhere.")
        return None
    print(f"{len(todo)} frame(s) where the staff is visible but the current run has no box on them:")
    print("  " + ", ".join(str(f) for f, _, _ in todo) + "\n")

    cap = open_video(video)
    imgs = {f: read_frame(cap, f) for f, _, _ in todo}
    cap.release()
    models, results = {}, []

    for name, w, sz, tta, rot in SWEEP:
        w = w or config.PERSON_WEIGHTS
        sz = sz or config.PERSON_IMGSZ
        try:
            if w not in models:
                models[w] = YOLO(w)
            model = models[w]
        except Exception as e:                         # e.g. a weights file that will not download
            print(f"{name:20s} skipped: {type(e).__name__}: {e}")
            continue
        found, boxes_seen, found_frames = 0, 0, []
        for f, x, y in todo:
            frame = imgs.get(f)
            if frame is None:
                continue
            H, W = frame.shape[:2]
            ok = False
            for code, fwd in (_ROT if rot else _ROT[:1]):
                img = frame if code is None else cv2.rotate(frame, code)
                px, py = fwd(x, y, W, H)
                r = model.predict(img, classes=[0], conf=conf, imgsz=sz,
                                  augment=tta, verbose=False)[0]
                boxes_seen += len(r.boxes)
                if _covers(r.boxes, px, py, pad):
                    ok = True
                    break
            found += ok
            if ok:
                found_frames.append(f)
        results.append((name, found, len(todo), boxes_seen, found_frames))
        print(f"  {name:20s} staff found in {found:>2}/{len(todo)} missed frames "
              f"({boxes_seen} boxes drawn in total)")

    if results:
        best = max(results, key=lambda r: (r[1], -r[3]))
        print(f"\nbest: {best[0]} -> {best[1]}/{best[2]} recovered; frames {best[4]}")
        print("A setting that recovers few or none means the overhead view defeats the pretrained "
              "model, and only fine-tuning on labelled overhead frames would help.")
    return results