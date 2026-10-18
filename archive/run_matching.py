import cv2
import pandas as pd
from pathlib import Path
from tag_matcher import build_template_bank, match_tag

# ---- Load templates ----
templates = {}
for p in sorted(Path("templates").glob("tag*.png")):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        print(f"Skipping unreadable template: {p}")
        continue
    templates[p.stem] = img
    print(f"Loaded {p.name}: {img.shape[1]}x{img.shape[0]} px")

if not templates:
    raise FileNotFoundError("No templates/tag*.png files found.")

bank = build_template_bank(templates)
print(f"Template bank size: {len(bank)}")

# ---- Load video and detections ----
cap = cv2.VideoCapture("videos/sample.mp4")
df = pd.read_csv("outputs/person_detections.csv")

# ---- Match every detection ----
rows = []
for frame_no, group in df.groupby("frame"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, frame = cap.read()
    if not ok:
        continue
    H, W = frame.shape[:2]

    for idx, r in group.iterrows():
        x1, y1 = max(0, int(r.x1)), max(0, int(r.y1))
        x2, y2 = min(W, int(r.x2)), min(H, int(r.y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # This line must be INSIDE the loop, after crop exists
        score, pt, tname, s, a = match_tag(crop, bank)
        rows.append({
            "det_idx": idx, "frame": int(frame_no), "score": score,
            "tag_x": x1 + pt[0] if pt else None,
            "tag_y": y1 + pt[1] if pt else None,
            "template": tname, "scale": s, "angle": a,
        })

cap.release()

# ---- Save ----
res = pd.DataFrame(rows).set_index("det_idx")
out = df.join(res[["score", "tag_x", "tag_y", "template", "scale", "angle"]])
out.to_csv("outputs/tag_scores.csv", index=False)

print(out["score"].describe())
print(out.sort_values("score", ascending=False).head(10)[["frame", "score", "template", "scale", "angle"]])