"""Run the trained drone detector on an image, a directory, or a video."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/drone-random/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")
    model = YOLO(args.weights)
    for _ in model.predict(
        source=args.source, conf=args.conf, imgsz=args.imgsz, device=args.device, save=True,
        stream=True, project=str(ROOT / "runs"), name="predict",
    ):
        pass


if __name__ == "__main__":
    main()
