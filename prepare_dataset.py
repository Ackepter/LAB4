"""Create reproducible YOLO split lists for the Roboflow export."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from random import Random
from shutil import copyfile
from statistics import median

from PIL import Image

DATA = Path(__file__).resolve().parent / "dataset"


def main():
    images = sorted((DATA / "train/images").iterdir())
    labels = DATA / "train/labels"
    output = DATA / "detection"
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    object_sizes = {}
    polygons = 0
    for image in images:
        label = labels / f"{image.stem}.txt"
        boxes = []
        areas = []
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, *coords = map(float, line.split())
            if len(coords) >= 6 and len(coords) % 2 == 0 and all(0 <= c <= 1 for c in coords):
                xs, ys = coords[::2], coords[1::2]
                coords = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                          max(xs) - min(xs), max(ys) - min(ys)]
                polygons += 1
            x, y, width, height = coords
            if not (class_id == 0 and 0 <= x <= 1 and 0 <= y <= 1
                    and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Invalid YOLO annotation in {label}: {line}")
            boxes.append(f"0 {x:.8f} {y:.8f} {width:.8f} {height:.8f}\n")
            areas.append(width * height)
        object_sizes[image] = median(areas) if areas else 0
        copyfile(image, output / "images" / image.name)
        (output / "labels" / label.name).write_text("".join(boxes), encoding="utf-8")
        with Image.open(image) as source:
            rgb = source.convert("RGB")
            digest = sha256(rgb.tobytes()).hexdigest()
        groups[rgb.size, digest].append(image)

    unique = sorted(groups.values(), key=lambda copies: object_sizes[copies[0]])
    splits = {name: [] for name in ("test", "val", "train")}
    random = Random(42)
    held_out = round(len(unique) * 0.15)
    for index in range(3):
        stratum = unique[index * len(unique) // 3:(index + 1) * len(unique) // 3]
        random.shuffle(stratum)
        count = (index + 1) * held_out // 3 - index * held_out // 3
        splits["test"].extend(stratum[:count])
        splits["val"].extend(stratum[count:2 * count])
        splits["train"].extend(stratum[2 * count:])
    for items in splits.values():
        random.shuffle(items)
    for split, items in splits.items():
        paths = [f"./detection/images/{path.name}" for path, *_ in items]
        (DATA / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        print(f"{split}: {len(paths)}")
        print(f"Median box area: {median(object_sizes[copies[0]] for copies in items):.2%}")
    print(f"Exact duplicates dropped: {len(images) - len(unique)}")
    print(f"Polygons converted to bounding boxes: {polygons}")


if __name__ == "__main__":
    main()
