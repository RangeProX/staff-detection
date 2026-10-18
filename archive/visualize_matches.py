import cv2
import numpy as np
import pandas as pd

df = pd.read_csv("outputs/tag_scores.csv").dropna(subset=["tag_x"])
cap = cv2.VideoCapture("videos/sample.mp4")

# One best detection per frame, then take the top 12 frames
best = df.sort_values("score", ascending=False).drop_duplicates("frame").head(12)

tiles = []
for _, r in best.iterrows():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
    ok, frame = cap.read()
    if not ok:
        continue
    cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 2)
    cv2.circle(frame, (int(r.tag_x), int(r.tag_y)), 8, (0, 0, 255), 2)
    cv2.putText(frame, f"f{int(r.frame)} s={r.score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    tiles.append(cv2.resize(frame, (480, 270)))

while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
cv2.imwrite("outputs/top_matches.jpg", np.vstack(rows))

# Best score per frame over time
per_frame = df.groupby("frame")["score"].max()
print(per_frame.describe())