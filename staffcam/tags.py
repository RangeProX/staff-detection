"""Stage 2: detect name tags on full frames -> outputs/<video>/tag_raw.csv"""
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from . import config
from .video import iter_frames

COLUMNS = ["frame", "conf", "tx1", "ty1", "tx2", "ty2", "tag_x", "tag_y"]


def run(video):
    video = Path(video)
    out = config.out_dir(video)
    tagger = YOLO(config.TAG_WEIGHTS)

    rows = []
    for f, frame in iter_frames(video):
        if f % config.TAG_STRIDE:
            continue
        res = tagger.predict(frame, conf=config.TAG_RAW_CONF,
                             imgsz=config.TAG_IMGSZ, verbose=False)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            rows.append({"frame": f, "conf": b.conf.item(),
                         "tx1": x1, "ty1": y1, "tx2": x2, "ty2": y2,
                         "tag_x": (x1 + x2) / 2, "tag_y": (y1 + y2) / 2})

    df = pd.DataFrame(rows, columns=COLUMNS)
    path = out / "tag_raw.csv"
    df.to_csv(path, index=False)
    print(f"{len(df)} raw tag detections -> {path}")
    return path
