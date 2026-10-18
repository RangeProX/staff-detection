"""Stage 3: tag-first staff identification and tracking.

A person becomes STAFF when a detected tag falls inside their box on the same tracker id
at least MIN_HITS times, counted as DISTINCT observations: two hits closer together than
MIN_GAP_SEC are one observation, because the scene has barely changed between them. Counting raw
hits instead would weaken the guard whenever the tag detector is run more often. From the first tag sighting on, that identity is followed forward.
Every tag-confirmed person gets their own identity (several staff can be present).

Sources (and tiers):
  tag_hit            tag detected inside this box in this frame                 -> confirmed
  same_track         confirmed tracker id, after its first hit, within TRUST_SEC of a hit -> confirmed
  track_before_tag   same tracker id BEFORE its first tag hit (BACKFILL_SEC)    -> inferred
  track_unsupported  confirmed tracker id, but far from any tag hit             -> inferred
  fill               followed after the tracker lost the id (see below)         -> inferred
  interpolated       straight line across a short gap with no detections        -> inferred

Following a lost identity:
  * the match gate is centred on the PREDICTED position (last velocity, extrapolated up to
    PREDICT_MAX_SEC) and its radius is capped at GATE_MAX person sizes;
  * size check: a box more than SIZE_RATIO times bigger or smaller than the person being followed
    is rejected (merged boxes covering several people and furniture);
  * handover rule: only a track that STARTED after the loss may take over, so a person who was
    already on screen (e.g. a stationary bystander) can never become staff. The exception is a
    track id this identity has already used (the tracker flickers between ids for one person).
All thresholds are relative to person size and time, not pixels or frame counts.
staff_events.csv logs starts, handovers and losses so failures can be diagnosed.
"""
import cv2
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

from . import config as C
from .video import open_video

HIT_COLS = ["frame", "track_id", "conf", "tag_x", "tag_y"]
OUT_COLS = ["frame", "timestamp", "staff_id", "track_id", "x1", "y1", "x2", "y2", "cx", "cy",
            "tag_x", "tag_y", "tag_conf", "x", "y", "source", "tier"]
EVENT_COLS = ["staff_id", "event", "frame", "detail", "value"]
CONFIRMED = {"tag_hit", "same_track"}

# Every linking setting this module reads, with the value to use if config.py lacks it.
DEFAULTS = {
    "CONF": 0.4, "MIN_HITS": 3, "MARGIN_FRAC": 0.1,
    "GATE_BASE": 0.3, "GATE_MAX": 1.0,
    "SPEED_MIN": 1.0, "SPEED_MAX": 6.0, "SPEED_PCT": 90,
    "PREDICT_MAX_SEC": 2.0, "MAX_MISS_SEC": 2.0, "MAX_INTERP_SEC": 1.0,
    "HANDOVER_TOL_SEC": 0.4, "TRUST_SEC": 3.0, "BACKFILL_SEC": 0.0, "SIZE_RATIO": 1.5,
    "MIN_GAP_SEC": 0.1,
}


def _check_config():
    """Fail up front, listing exactly which lines to add to config.py."""
    missing = {k: v for k, v in DEFAULTS.items() if not hasattr(C, k)}
    if missing:
        lines = "\n".join(f"{k} = {v}" for k, v in missing.items())
        raise RuntimeError("config.py is missing linking settings. Add these lines:\n\n" + lines + "\n")


def _gate(size, dt_sec, speed):
    """Match radius in px: grows with time since last seen, capped at GATE_MAX person sizes."""
    return size * min(C.GATE_BASE + speed * dt_sec, C.GATE_MAX)


def _speed(anch, fps):
    """Walking speed of this video, in person sizes per second: a high percentile of the speed of
    tag-confirmed boxes, measured over 5-frame windows. Uses no hand labels."""
    v = []
    for _, g in anch.groupby("track_id"):
        g = g.sort_values("frame")
        if len(g) <= 5:
            continue
        f, cx, cy, sz = g.frame.values, g.cx.values, g.cy.values, g.psize.values
        d = np.hypot(cx[5:] - cx[:-5], cy[5:] - cy[:-5])
        ok = (f[5:] - f[:-5]) == 5
        v += list(d[ok] / sz[:-5][ok] * fps / 5)
    if len(v) < 20:
        return C.SPEED_MIN
    return float(np.clip(np.percentile(v, C.SPEED_PCT), C.SPEED_MIN, C.SPEED_MAX))


def _distinct_hits(frames, fps):
    """How many DISTINCT observations these hit frames represent: walking a sorted pass and skipping
    any hit within MIN_GAP_SEC of the last one counted. Independent of how often the detector runs."""
    gap = C.MIN_GAP_SEC * fps
    n, last = 0, None
    for f in np.sort(np.unique(np.asarray(frames, dtype=float))):
        if last is None or f - last >= gap:
            n += 1
            last = f
    return n


def _tag_hits(raw, tracked_by_frame):
    """Assign each tag detection to the smallest tracked person box that contains it."""
    hits = []
    for t in raw.itertuples():
        p = tracked_by_frame.get(int(t.frame))
        if p is None:
            continue
        m = C.MARGIN_FRAC * p.psize
        p = p[(p.x1 - m <= t.tag_x) & (t.tag_x <= p.x2 + m)
              & (p.y1 - m <= t.tag_y) & (t.tag_y <= p.y2 + m)]
        if len(p):
            p = p.assign(a=(p.x2 - p.x1) * (p.y2 - p.y1)).nsmallest(1, "a").iloc[0]
            hits.append({"frame": int(t.frame), "track_id": int(p.track_id),
                         "conf": t.conf, "tag_x": t.tag_x, "tag_y": t.tag_y})
    return pd.DataFrame(hits, columns=HIT_COLS)


def _row(b, sid, source, hit):
    tx, ty, tc = hit if hit is not None else (np.nan, np.nan, np.nan)
    return dict(frame=int(b.frame), timestamp=b.timestamp, staff_id=sid, track_id=int(b.track_id),
                x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, cx=b.cx, cy=b.cy,
                tag_x=tx, tag_y=ty, tag_conf=tc,
                x=b.cx if hit is None else tx, y=b.cy if hit is None else ty,
                source=source, tier="confirmed" if source in CONFIRMED else "inferred")


def _interpolate(rows, fps, speed):
    """Straight-line fill of short gaps, per identity, only if the two ends could be one walk."""
    by = {}
    for r in rows:
        by.setdefault(r["staff_id"], []).append(r)
    new = []
    for sid, rs in by.items():
        rs.sort(key=lambda r: r["frame"])
        for a, b in zip(rs, rs[1:]):
            gap = b["frame"] - a["frame"] - 1
            if not 0 < gap <= C.MAX_INTERP_SEC * fps:
                continue
            size = max(a["x2"] - a["x1"], a["y2"] - a["y1"])
            size_b = max(b["x2"] - b["x1"], b["y2"] - b["y1"])
            if max(size / size_b, size_b / size) > C.SIZE_RATIO:
                continue
            if np.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"]) > size * (C.GATE_BASE + speed * (gap + 1) / fps):
                continue
            for f in range(a["frame"] + 1, b["frame"]):
                t = (f - a["frame"]) / (b["frame"] - a["frame"])
                lin = {k: a[k] + t * (b[k] - a[k]) for k in ("x1", "y1", "x2", "y2", "cx", "cy", "timestamp")}
                new.append(dict(lin, frame=f, staff_id=sid, track_id=-1, tag_x=np.nan, tag_y=np.nan,
                                tag_conf=np.nan, x=lin["cx"], y=lin["cy"],
                                source="interpolated", tier="inferred"))
    return new


JUMP_MULT = 1.5      # a labelled box may move up to this many times the measured walking speed


def _jumps(staff, fps, speed):
    """Places where an identity's labelled box moved faster than a person can: the usual sign of
    a label that switched to another person (ID swap, merged box, wrong handover)."""
    out = []
    for sid, g in staff.groupby("staff_id"):
        g = g.sort_values("frame")
        f, cx, cy = g.frame.values, g.cx.values, g.cy.values
        sz = np.maximum(g.x2 - g.x1, g.y2 - g.y1).values
        for k in range(1, len(g)):
            dt = (f[k] - f[k - 1]) / fps
            if dt > C.MAX_MISS_SEC:
                continue
            moved = float(np.hypot(cx[k] - cx[k - 1], cy[k] - cy[k - 1]) / sz[k - 1])
            limit = C.GATE_BASE + JUMP_MULT * speed * dt
            if moved > limit:
                out.append((int(sid), int(f[k - 1]), int(f[k]), moved, limit))
    return out


def _diagnose(raw_all, raw, hits, counts):
    print("NO STAFF IDENTIFIED. Nothing was lowered or guessed. Diagnostics:")
    mx = raw_all.conf.max() if len(raw_all) else float("nan")
    print(f"  raw tag detections: {len(raw_all)} (max confidence {mx:.2f})")
    print(f"  at or above CONF={C.CONF}: {len(raw)}")
    print(f"  of those, inside a tracked person box: {len(hits)}")
    print(f"  most DISTINCT observations on a single track: {int(counts.max()) if len(counts) else 0} "
          f"(MIN_HITS={C.MIN_HITS}, MIN_GAP_SEC={C.MIN_GAP_SEC}s)")


def run(video):
    _check_config()
    out = C.out_dir(Path(video))
    cap = open_video(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    path = out / "staff_positions.csv"

    raw_all = pd.read_csv(out / "tag_raw.csv")
    raw = raw_all[raw_all.conf >= C.CONF]
    det = pd.read_csv(out / "person_detections.csv")
    if det.empty:
        pd.DataFrame(columns=OUT_COLS).to_csv(path, index=False)
        print("No person detections.")
        return path
    det["cx"] = (det.x1 + det.x2) / 2
    det["cy"] = (det.y1 + det.y2) / 2
    det["psize"] = np.maximum(det.x2 - det.x1, det.y2 - det.y1)
    tracked = det[det.track_id >= 0]
    det_by_frame = {int(f): g for f, g in det.groupby("frame")}

    # 1. tag hits -> person tracks -> confirmed staff tracks
    hits = _tag_hits(raw, {int(f): g for f, g in tracked.groupby("frame")})
    hits.to_csv(out / "tag_hits.csv", index=False)
    counts = (pd.Series({int(t): _distinct_hits(g.frame.values, fps) for t, g in hits.groupby("track_id")},
                        dtype=int) if len(hits) else pd.Series(dtype=int))
    confirmed = counts[counts >= C.MIN_HITS].index
    collapsed = [(int(t), len(g), int(counts.get(int(t), 0))) for t, g in hits.groupby("track_id")
                 if len(g) >= C.MIN_HITS > counts.get(int(t), 0)]
    if len(confirmed) == 0:
        pd.DataFrame(columns=OUT_COLS).to_csv(path, index=False)
        pd.DataFrame(columns=EVENT_COLS).to_csv(out / "staff_events.csv", index=False)
        _diagnose(raw_all, raw, hits, counts)
        return path
    ch = hits[hits.track_id.isin(confirmed)]
    first_hit = ch.groupby("track_id").frame.min().to_dict()
    hit_frames = {int(t): np.sort(g.frame.unique()) for t, g in ch.groupby("track_id")}
    best = hits.sort_values("conf", ascending=False).drop_duplicates(["frame", "track_id"])
    hit_info = {(int(r.frame), int(r.track_id)): (r.tag_x, r.tag_y, r.conf) for r in best.itertuples()}
    start = tracked.groupby("track_id").frame.min().to_dict()

    anch = tracked[tracked.track_id.isin(first_hit)]
    # labelled from the first tag sighting, or BACKFILL_SEC earlier on the same tracker id
    anch = anch[anch.frame >= anch.track_id.map(first_hit) - C.BACKFILL_SEC * fps]
    anch_by_frame = {int(f): g for f, g in anch.groupby("frame")}

    speed = _speed(anch, fps)
    tol, max_miss, trust = C.HANDOVER_TOL_SEC * fps, C.MAX_MISS_SEC * fps, C.TRUST_SEC * fps
    identities, rows, events, next_id = [], [], [], 1

    def near_hit(tid, f):
        a = hit_frames[tid]
        return np.abs(a - f).min() <= trust

    def gate(i, f):
        return _gate(i["pos"][2], (f - i["last_f"]) / fps, speed)

    def predicted(i, f):
        """Last known centre, moved along the recent velocity for up to PREDICT_MAX_SEC."""
        cx, cy = i["pos"][0], i["pos"][1]
        h = i["hist"]
        if len(h) >= 3 and h[-1][0] - h[0][0] >= 3:
            span = h[-1][0] - h[0][0]
            dt = min(f - i["last_f"], C.PREDICT_MAX_SEC * fps)
            return cx + dt * (h[-1][1] - h[0][1]) / span, cy + dt * (h[-1][2] - h[0][2]) / span
        return cx, cy

    def dist(i, b, f):
        px, py = predicted(i, f)
        return float(np.hypot(b.cx - px, b.cy - py))

    def size_ok(i, b):
        """A box much bigger or smaller than the person being followed is not that person
        (e.g. a merged box covering several people and furniture)."""
        r = b.psize / i["pos"][2]
        return max(r, 1 / r) <= C.SIZE_RATIO

    def follow(i, b, f):
        old = i["pos"]
        if old is not None and old[3] != int(b.track_id):
            perr = dist(i, b, f) / old[2]
            moved = np.hypot(b.cx - old[0], b.cy - old[1]) / old[2]
            events.append(dict(staff_id=i["id"], event="handover", frame=f, value=perr,
                               detail=f"track {old[3]}->{int(b.track_id)} gap={f - i['last_f']}f "
                                      f"moved={moved:.2f} sizes, off prediction by {perr:.2f} sizes"))
        i["pos"], i["last_f"] = (b.cx, b.cy, b.psize, int(b.track_id)), f
        i["hist"].append((f, b.cx, b.cy))
        i["tracks"].add(int(b.track_id))

    def lost(i, f_end):
        """Why did this identity end? Nearest tracked box to its PREDICTED position, new vs pre-existing."""
        size, tid = i["pos"][2], i["pos"][3]
        best = {"new": None, "old": None}
        for g in range(i["last_f"] + 1, f_end + 1):
            d = det_by_frame.get(g)
            if d is None:
                continue
            d = d[d.track_id >= 0]
            if d.empty:
                continue
            px, py = predicted(i, g)
            dd = np.hypot(d.cx - px, d.cy - py) / size
            for t, v in zip(d.track_id, dd):
                kind = "new" if start.get(int(t), g) >= i["last_f"] - tol else "old"
                if best[kind] is None or v < best[kind][0]:
                    best[kind] = (float(v), g, int(t))
        fmt = lambda x: "none in window" if x is None else f"{x[0]:.2f} sizes (track {x[2]}, frame {x[1]})"
        events.append(dict(staff_id=i["id"], event="lost", frame=i["last_f"],
                           detail=f"last track {tid}; no match for {(f_end - i['last_f']) / fps:.1f}s; "
                                  f"nearest NEW track to predicted position: {fmt(best['new'])}; "
                                  f"nearest PRE-EXISTING: {fmt(best['old'])}; gate cap {C.GATE_MAX} sizes"))

    # 2. forward pass: tracking starts at the first tag sighting of each person
    for f in range(total):
        taken, updated = set(), set()
        active = [i for i in identities if i["active"]]

        for b in (anch_by_frame[f].itertuples() if f in anch_by_frame else []):
            tid = int(b.track_id)
            hit = hit_info.get((f, tid))
            ident = next((i for i in active if tid in i["tracks"] and i["id"] not in updated), None)
            if ident is not None and not hit and not size_ok(ident, b):
                continue                           # box ballooned (merged with others): don't follow it
            if ident is None:                      # newly confirmed track: handover, else a new person
                cands = [i for i in active if i["id"] not in updated
                         and start.get(tid, f) >= i["last_f"] - tol
                         and dist(i, b, f) <= gate(i, f)]
                if cands:
                    ident = min(cands, key=lambda i: dist(i, b, f))
                else:
                    ident = {"id": next_id, "pos": None, "last_f": f, "active": True, "hist": deque(maxlen=8), "tracks": set()}
                    events.append(dict(staff_id=next_id, event="start", frame=f, detail=f"track {tid}"))
                    next_id += 1
                    identities.append(ident)
            if hit:
                source = "tag_hit"
            elif f < first_hit[tid]:
                source = "track_before_tag"        # same id as a confirmed track, but before its first tag hit
            else:
                source = "same_track" if near_hit(tid, f) else "track_unsupported"
            rows.append(_row(b, ident["id"], source, hit))
            follow(ident, b, f)
            taken.add(b.Index)
            updated.add(ident["id"])

        boxes = det_by_frame.get(f)
        if boxes is not None:
            free = boxes[(boxes.track_id >= 0) & ~boxes.index.isin(taken)]
            props = []
            for i in active:
                if i["id"] in updated:
                    continue
                g = gate(i, f)
                for b in free.itertuples():
                    if not size_ok(i, b):
                        continue
                    d = dist(i, b, f)
                    if b.track_id == i["pos"][3]:
                        if d <= 2 * g:
                            props.append((0, d, i, b))
                    elif b.track_id in i["tracks"]:
                        if d <= 2 * g:
                            props.append((1, d, i, b))   # a track id this person already used (ids flicker)
                    elif start.get(b.track_id, f) >= i["last_f"] - tol and d <= g:
                        props.append((2, d, i, b))       # handover to a track that started after the loss
            for _, _, i, b in sorted(props, key=lambda p: (p[0], p[1])):
                if i["id"] in updated or b.Index in taken:
                    continue
                rows.append(_row(b, i["id"], "fill", None))
                follow(i, b, f)
                taken.add(b.Index)
                updated.add(i["id"])

        for i in active:
            if i["id"] not in updated and f - i["last_f"] > max_miss:
                i["active"] = False
                lost(i, f)

    rows += _interpolate(rows, fps, speed)
    staff = pd.DataFrame(rows, columns=OUT_COLS).sort_values(["frame", "staff_id"])
    staff[["frame", "staff_id", "track_id"]] = staff[["frame", "staff_id", "track_id"]].astype(int)
    staff.to_csv(path, index=False)
    for sid, f0, f1, moved, limit in _jumps(staff, fps, speed):
        events.append(dict(staff_id=sid, event="jump", frame=f1, value=moved,
                           detail=f"labelled box moved {moved:.2f} sizes between frames {f0} and {f1} "
                                  f"(plausible up to {limit:.2f})"))
    ev = pd.DataFrame(events, columns=EVENT_COLS)
    ev.to_csv(out / "staff_events.csv", index=False)

    n = staff.staff_id.nunique()
    print(f"{n} staff identit{'y' if n == 1 else 'ies'} (CONF={C.CONF}, MIN_HITS={C.MIN_HITS})")
    for sid, g in staff.groupby("staff_id"):
        print(f"  staff {sid}: frames {g.frame.min()}-{g.frame.max()}, "
              f"{g.frame.nunique()} labelled, sources: {g.source.value_counts().to_dict()}")
    print(f"walking speed (p{C.SPEED_PCT} of tag-confirmed boxes, measured): {speed:.1f} person sizes/s")
    if collapsed:
        print(f"  {len(collapsed)} track(s) reached {C.MIN_HITS} tag hits but too few distinct observations "
              f"(MIN_GAP_SEC={C.MIN_GAP_SEC}s), so were NOT confirmed: "
              + ", ".join(f"track {t} ({n} hits -> {d} distinct)" for t, n, d in collapsed))
    ids = sorted(staff.staff_id.unique())
    fr = {sid: set(staff[staff.staff_id == sid].frame) for sid in ids}
    for k, a in enumerate(ids):
        for b in ids[k + 1:]:
            both = fr[a] & fr[b]
            if both:
                print(f"  note: staff {a} and {b} overlap in time (frames {min(both)}-{max(both)}, {len(both)} frames); "
                      "expected only if two tagged people were present at once")
    tier = staff.groupby("tier").frame.nunique().to_dict()
    print(f"frames with staff: confirmed={tier.get('confirmed', 0)}, inferred-only={tier.get('inferred', 0)} "
          f"(video has {total})")
    print(f"events -> {out / 'staff_events.csv'}")
    jumps = ev[ev.event == "jump"]
    if len(jumps):
        print(f"  {len(jumps)} JUMP(S): the labelled box moved faster than a person can, "
              "which usually means the label switched to someone else:")
        for r in jumps.head(12).itertuples():
            print(f"  [staff {r.staff_id}] JUMP at frame {r.frame}: {r.detail}")
    for r in ev[ev.event == "lost"].itertuples():
        print(f"  [staff {r.staff_id}] LOST after frame {r.frame}: {r.detail}")
    for r in ev[ev.event == "handover"].itertuples():
        if r.value > 0.5:
            print(f"  [staff {r.staff_id}] CHECK handover at frame {r.frame}: {r.detail}")
    return path