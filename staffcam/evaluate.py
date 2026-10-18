"""Ground truth + scoring: how often is each kind of label actually on the staff member?

  python run.py annotate --video ... --step 15   click the staff in sampled frames (resumable)
  python run.py score    --video ...             precision per label source, recall, false alarms
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from . import config as C
from .video import open_video, read_frame


def annotate(video, step=15, offset=0, name="ground_truth"):
    out = C.out_dir(Path(video))
    path = out / f"{name}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    seen = {int(r["frame"]) for r in rows}
    cap = open_video(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    click = {}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    cv2.namedWindow("ground truth")
    cv2.setMouseCallback("ground truth", on_mouse)
    for f in range(offset, total, step):
        if f in seen:
            continue
        frame = read_frame(cap, f)
        if frame is None:
            break
        scale = min(1.0, 1400 / frame.shape[1])
        base = cv2.resize(frame, None, fx=scale, fy=scale)
        click.clear()
        done = False
        while not done:
            view = base.copy()
            if "pt" in click:
                cv2.circle(view, click["pt"], 8, (0, 0, 255), 2)
            cv2.putText(view, f"frame {f}: click the staff, ENTER=save  n=staff not visible  s=skip  q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("ground truth", view)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 32) and "pt" in click:
                rows.append(dict(frame=f, visible=1, x=click["pt"][0] / scale, y=click["pt"][1] / scale))
                done = True
            elif k == ord("n"):
                rows.append(dict(frame=f, visible=0, x=np.nan, y=np.nan))
                done = True
            elif k == ord("s"):
                done = True
            elif k == ord("q"):
                pd.DataFrame(rows).to_csv(path, index=False)
                cap.release()
                cv2.destroyAllWindows()
                print(f"Saved {len(rows)} annotations -> {path}")
                return path
        pd.DataFrame(rows).to_csv(path, index=False)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(rows)} annotations -> {path}")
    return path


def score(video, pad=0.1, bucket=150, frm=0, to=None, name="ground_truth"):
    """A label is correct if the clicked point lies inside the labelled box (padded by `pad` sizes).
    Missed frames are split by whether the person detector had ANY box on the clicked person."""
    out = C.out_dir(Path(video))
    gt_path = out / f"{name}.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"No {gt_path.name}. Run `annotate` first"
                                + (f' with --gt-name {name}' if name != "ground_truth" else "") + ".")
    print(f"ground truth: {gt_path.name}")
    gt = pd.read_csv(gt_path)
    gt = gt[(gt.frame >= frm) & (gt.frame <= (to if to is not None else gt.frame.max()))]
    if gt.empty:
        raise RuntimeError(f"No ground-truth frames in range {frm}-{to}.")
    staff = pd.read_csv(out / "staff_positions.csv")
    det = pd.read_csv(out / "person_detections.csv")
    by = {int(f): g for f, g in staff.groupby("frame")}
    det_by = {int(f): g for f, g in det.groupby("frame")}

    def boxes_on(f, x, y):
        g = det_by.get(f)
        if g is None:
            return []
        m = pad * np.maximum(g.x2 - g.x1, g.y2 - g.y1)
        return g[(g.x1 - m <= x) & (x <= g.x2 + m) & (g.y1 - m <= y) & (y <= g.y2 + m)].track_id.tolist()

    pairs, frames, absent, absent_fp = [], [], 0, 0
    for r in gt.itertuples():
        f = int(r.frame)
        g = by.get(f)
        if not r.visible:
            absent += 1
            absent_fp += g is not None
            continue
        hit = False
        if g is not None:
            for s_ in g.itertuples():
                m = pad * max(s_.x2 - s_.x1, s_.y2 - s_.y1)
                ok = (s_.x1 - m <= r.x <= s_.x2 + m) and (s_.y1 - m <= r.y <= s_.y2 + m)
                pairs.append(dict(source=s_.source, tier=s_.tier, correct=ok))
                hit |= ok
        tracks = boxes_on(f, r.x, r.y)
        frames.append(dict(frame=f, labelled=g is not None, correct=hit, detected=len(tracks) > 0, tracks=tracks))

    v = pd.DataFrame(frames)
    n = len(v)
    print(f"annotated frames {int(gt.frame.min())}-{int(gt.frame.max())}: "
          f"{n} with the staff visible, {absent} marked not visible")
    if n:
        missed = v[~v.correct & ~v.labelled]
        wrong = v[v.labelled & ~v.correct]
        print(f"  recall (staff box on the clicked person): {v.correct.sum()}/{n} = {v.correct.mean():.0%}")
        print(f"  person detector has a box on the staff:   {v.detected.sum()}/{n} = {v.detected.mean():.0%}"
              "   <- ceiling for linking (interpolation can exceed it)")
        print(f"  wrong person labelled: {len(wrong)}   missed (no label): {len(missed)}")
        link_miss = missed[missed.detected]
        det_miss = missed[~missed.detected]
        print(f"    missed WITH a person box on the staff (linking problem):  {len(link_miss)}")
        print(f"      frames {link_miss.frame.tolist()[:15]}")
        print(f"      track ids there: {sorted({t for ts in link_miss.tracks for t in ts})[:15]}")
        print(f"    missed with NO person box on the staff (detector problem): {len(det_miss)}")
        print(f"      frames {det_miss.frame.tolist()[:15]}")
        if len(wrong):
            print("  wrong-person frames:", wrong.frame.tolist()[:20])
        v["b"] = (v.frame // bucket) * bucket
        t = v.groupby("b").agg(visible=("frame", "size"), correct=("correct", "sum"), has_box=("detected", "sum"))
        t.index = [f"{b}-{b + bucket - 1}" for b in t.index]
        print(f"\nby time ({bucket}-frame buckets):\n{t.to_string()}")
    if absent:
        print(f"\nfalse alarms on 'not visible' frames: {absent_fp}/{absent}")
    if pairs:
        p = pd.DataFrame(pairs)
        for key in ("source", "tier"):
            t = p.groupby(key).correct.agg(n="size", correct="sum")
            t["precision"] = (t.correct / t.n).round(2)
            print(f"\nprecision by {key}:\n{t.to_string()}")
    print("\nNote: one click per frame, so a second staff member's box counts as 'wrong'.")
    return v