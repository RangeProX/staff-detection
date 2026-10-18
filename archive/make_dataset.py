import cv2, pandas as pd
from pathlib import Path

LO, HI = 810, 1254          # frames where you know the tag is visible
MIN_SCORE = 0.85
BOX = 40                    # tag box side in px; set from your template sizes

df = pd.read_csv("outputs/tag_scores.csv").dropna(subset=["tag_x"])
df = df[df["frame"].between(LO, HI) & (df["score"] >= MIN_SCORE)]
# skip the frames your templates were cut from, and their neighbours
for f in (465, 1020):
    df = df[~df["frame"].between(f - 15, f + 15)]
df = df.sort_values("score", ascending=False).drop_duplicates("frame")

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    Path(f"dataset/{sub}").mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture("videos/sample.mp4")
kept = 0
for _, r in df.sort_values("frame").iterrows():
    f = int(r.frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        continue
    H, W = frame.shape[:2]
    cx, cy = int(r.tag_x), int(r.tag_y)
    x1, y1 = max(0, cx - BOX // 2), max(0, cy - BOX // 2)
    x2, y2 = min(W, cx + BOX // 2), min(H, cy + BOX // 2)

    view = frame.copy()
    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 1)
    zx1, zy1 = max(0, cx - 100), max(0, cy - 100)
    zoom = cv2.resize(view[zy1:cy + 100, zx1:cx + 100], None, fx=4, fy=4)
    cv2.imshow(f"frame {f}  y=keep n=skip q=quit", zoom)
    k = cv2.waitKey(0) & 0xFF
    if k == ord("q"):
        break
    if k != ord("y"):
        continue

    # time-based split: last 20% of the range is validation
    split = "val" if f > LO + 0.8 * (HI - LO) else "train"
    cv2.imwrite(f"dataset/images/{split}/f{f}.jpg", frame)
    with open(f"dataset/labels/{split}/f{f}.txt", "w") as fh:
        fh.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
    kept += 1
print("kept", kept)