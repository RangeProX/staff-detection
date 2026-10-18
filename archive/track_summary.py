import pandas as pd

df = pd.read_csv("outputs/tag_scores.csv")
df = df[(df["score"] > -1) & (df["track_id"] >= 0)]

# Drop the frames the templates were cut from, plus their neighbours
for f in (465, 1020):
    df = df[~df["frame"].between(f - 15, f + 15)]

summary = df.groupby("track_id").agg(
    first_frame=("frame", "min"),
    last_frame=("frame", "max"),
    n_scored=("frame", "count"),
    top5_mean=("score", lambda s: s.nlargest(5).mean()),
    max_score=("score", "max"),
).sort_values("top5_mean", ascending=False)

print(summary.to_string())