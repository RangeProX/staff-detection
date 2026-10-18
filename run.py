import argparse
from staffcam import people, tags, link, viz, evaluate

PIPELINE = {"people": people.run, "tags": tags.run, "link": link.run}


def main():
    ap = argparse.ArgumentParser(description="Staff identification pipeline")
    ap.add_argument("command", choices=[*PIPELINE, "all", "render", "timeline", "sheet", "probe", "sweep", "annotate", "score"])
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=0, help="sheet: first frame")
    ap.add_argument("--end", type=int, default=300, help="sheet: last frame")
    ap.add_argument("--step", type=int, default=15, help="sheet/probe: frame step")
    ap.add_argument("--offset", type=int, default=0, help="annotate: first frame (use with --step for a disjoint holdout set)")
    ap.add_argument("--gt-name", default="ground_truth", help="annotate/score: ground-truth file name, without .csv")
    ap.add_argument("--gt-from", type=int, default=0, help="score: first ground-truth frame to include")
    ap.add_argument("--gt-to", type=int, default=None, help="score: last ground-truth frame to include")
    ap.add_argument("--weights", default=None, help="probe: person model, e.g. yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=None, help="probe: inference size")
    ap.add_argument("--conf", type=float, default=None, help="probe: confidence floor (default 0.1)")
    a = ap.parse_args()

    if a.command == "render":
        viz.render(a.video)
    elif a.command == "timeline":
        viz.timeline(a.video)
    elif a.command == "sheet":
        viz.sheet(a.video, a.start, a.end, a.step)
    elif a.command == "annotate":
        evaluate.annotate(a.video, a.step, a.offset, a.gt_name)
    elif a.command == "score":
        evaluate.score(a.video, frm=a.gt_from, to=a.gt_to, name=a.gt_name)
    elif a.command == "sweep":
        people.sweep(a.video, a.conf)
    elif a.command == "probe":
        people.probe(a.video, a.weights, a.imgsz, a.conf, a.start, a.end, a.step)
    else:
        for name in (PIPELINE if a.command == "all" else [a.command]):
            print(f"\n=== {name} ===")
            PIPELINE[name](a.video)


if __name__ == "__main__":
    main()