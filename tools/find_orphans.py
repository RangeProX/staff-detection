# find_orphans.py
from pathlib import Path

for split in ("train", "val"):
    imgs = {p.stem for p in Path(f"dataset/images/{split}").glob("*.jpg")}
    lbls = {p.stem for p in Path(f"dataset/labels/{split}").glob("*.txt")}
    print(split, "images without labels:", sorted(imgs - lbls))
    print(split, "labels without images:", sorted(lbls - imgs))