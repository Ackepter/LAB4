"""Fine-tune YOLOv8s and plot metrics on validation and test splits."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset/data.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))
    from ultralytics import YOLO, settings

    if settings["sync"]:
        settings.update({"sync": False})
    model = YOLO(ROOT / "yolov8s.pt")
    model.train(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, seed=42, cache=False, project=str(ROOT / "runs"),
        name="drone-random", plots=True, amp=False, close_mosaic=5,
    )
    run = Path(model.trainer.save_dir)
    model = YOLO(run / "weights/best.pt")
    metrics = model.val(
        data=str(DATA), split="test", imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, plots=True, project=str(run), name="test",
    )
    scores = {
        "mAP@50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    (run / "test_metrics.json").write_text(
        json.dumps({**scores, "imgsz": args.imgsz}, indent=2) + "\n", encoding="utf-8"
    )
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax.bar(list(scores), list(scores.values()), color="#286c9c", width=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(ylim=(0, 1.08), ylabel="Значение метрики",
           title="YOLOv8s · оценка на 51 тестовом изображении")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(run / "test_metrics.png", dpi=180)
    plt.close(fig)

    test_images = (DATA.parent / "test.txt").read_text(encoding="utf-8").splitlines()
    examples = test_images[:5]
    model.predict(
        source=[str(DATA.parent / name) for name in examples], imgsz=args.imgsz,
        conf=0.25, device=args.device, save=True, project=str(run), name="examples",
    )


if __name__ == "__main__":
    main()
