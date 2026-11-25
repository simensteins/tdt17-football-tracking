"""Fine-tune RF-DETR on the YOLO-format dataset written by make_yolo_dataset.py.

RF-DETR has its own dependencies, so it runs in a separate environment:

    python -m venv ~/venvs/rfdetr && source ~/venvs/rfdetr/bin/activate
    pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install rfdetr==1.12.1

    python scripts/train_rfdetr.py /cluster/work/$USER/yolo/stride5 --size medium --resolution 1280 \
        --epochs 10 --out /cluster/work/$USER/runs/rfdetr-m-1280

RF-DETR resizes frames to a square without keeping the aspect ratio, so at
1280 a 16 px ball becomes about 11 x 19 px. Its default scale jitter crops
frames to 384-600 px first, which would shrink the ball to 2-3 px, so it is
turned off. Writes checkpoint_best_total.pth and metrics.csv to --out.

--hue H shifts all hues of a training image together by up to H of the colour
wheel, like Ultralytics' hsv_h (needs rfdetr[augment] for Albumentations).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="folder with data.yaml, images/ and labels/")
    parser.add_argument("--size", choices=("nano", "small", "medium", "large"), default="medium")
    parser.add_argument("--resolution", type=int, default=1280, help="multiple of 32")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch x grad-accum")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hue", type=float, default=0.0, help="hue shift as a fraction of the colour wheel, e.g. 0.5")
    parser.add_argument("--freeze-encoder", action="store_true", help="keep the DINOv2 backbone fixed")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=Path, help="last.ckpt of an interrupted run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import rfdetr

    model_class = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
                   "medium": rfdetr.RFDETRMedium, "large": rfdetr.RFDETRLarge}[args.size]
    model = model_class(freeze_encoder=args.freeze_encoder)
    augment = {}
    if args.hue:  # RF-DETR's default is the horizontal flip alone; keep it and add the hue shift
        augment["aug_config"] = {
            "HorizontalFlip": {"p": 0.5},
            "HueSaturationValue": {"hue_shift_limit": round(args.hue * 180), "sat_shift_limit": 0,
                                   "val_shift_limit": 0, "p": 1.0},  # OpenCV hue runs 0-180
        }
    model.train(
        dataset_dir=str(args.dataset), dataset_file="yolo", output_dir=str(args.out),
        resolution=args.resolution, epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum,
        lr=args.lr, num_workers=args.workers, seed=0, device="cuda",
        scale_jitter=False, log_per_class_metrics=True, checkpoint_interval=1000,
        **augment, **({"resume": str(args.resume)} if args.resume else {}),
    )


if __name__ == "__main__":
    main()
