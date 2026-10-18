"""All tunable settings and path helpers live here. Change values here, not in the modules."""
from pathlib import Path

# ---- Models ----
PERSON_WEIGHTS = "yolo11n.pt"
TAG_WEIGHTS = "models/tag_v4.pt"

# ---- Person detection / tracking ----
PERSON_CONF = 0.1
PERSON_IMGSZ = 960
TRACKER = "bytetrack.yaml"
PREVIEW_EVERY_SEC = 2          # annotated preview JPG interval (0 = none)

# ---- Tag detection ----
TAG_RAW_CONF = 0.05            # floor when saving raw detections; real cutoff is CONF below
TAG_IMGSZ = 960
MIN_GAP_SEC = 0.1              # tag hits closer together than this count as one observation
TAG_STRIDE = 1                 # run the tag detector on every Nth frame (1 = every frame)

# ---- Tag -> person linking ----
CONF = 0.4
MIN_HITS = 3
MARGIN_FRAC = 0.1          # tag may sit this fraction of person size outside the box
GATE_BASE = 0.3            # match radius = person size * (GATE_BASE + GATE_SPEED * seconds since last seen)
SPEED_MIN = 1.0
SPEED_MAX = 6.0
SPEED_PCT = 90
BACKFILL_SEC = 0.0
SIZE_RATIO = 1.5
GATE_MAX = 1.0             # cap on the match radius, in person sizes
PREDICT_MAX_SEC = 1.0      # extrapolate the last velocity for up to this long
MAX_MISS_SEC = 2.0         # stop following an identity after this long without a match
TRUST_SEC = 3.0            # a tracker id counts as confirmed only this close to a tag hit
MAX_INTERP_SEC = 0.6       # straight-line fill for no-detection gaps up to this long
HANDOVER_TOL_SEC = 0.4     # a successor track may start this long before the old one was last seen


def out_dir(video) -> Path:
    """outputs/<video name>/ so a second video never overwrites the first."""
    d = Path("outputs") / Path(video).stem
    d.mkdir(parents=True, exist_ok=True)
    return d
