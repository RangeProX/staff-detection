import cv2
from pathlib import Path

VIDEO = "videos/sample.mp4"
STRIDE = 6
VAL_RANGE = (1100, 1300)       # contiguous block held out for validation

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    Path(f"dataset/{sub}").mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n_pos = n_neg = 0

for f in range(0, total, STRIDE):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        break
    H, W = frame.shape[:2]
    scale = min(1.0, 1400 / W)
    view = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.putText(view, f"frame {f}  l=label tag  n=no tag  other=skip  q=quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("label", view)
    k = cv2.waitKey(0) & 0xFF
    if k == ord("q"):
        break
    if k not in (ord("l"), ord("n")):
        continue

    split = "val" if VAL_RANGE[0] <= f <= VAL_RANGE[1] else "train"
    label_path = Path(f"dataset/labels/{split}/f{f}.txt")

    if k == ord("l"):
        x, y, w, h = cv2.selectROI("label", view, showCrosshair=False)
        if w == 0 or h == 0:
            continue
        x, y, w, h = x / scale, y / scale, w / scale, h / scale
        label_path.write_text(
            f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
        n_pos += 1
    else:
        label_path.write_text("")          # empty file = negative image
        n_neg += 1
    cv2.imwrite(f"dataset/images/{split}/f{f}.jpg", frame)

print("positives:", n_pos, "negatives:", n_neg)
cv2.destroyAllWindows()