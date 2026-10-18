from pathlib import Path

for split in ("train", "val"):
    files = list(Path(f"dataset/labels/{split}").glob("*.txt"))
    pos = sum(1 for f in files if f.read_text().strip())
    print(f"{split}: {len(files)} images, {pos} positives, {len(files) - pos} negatives")