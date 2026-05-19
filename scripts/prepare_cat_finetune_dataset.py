#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cat_cannon.app.finetune_dataset import prepare_finetune_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a reviewed cat dataset in Ultralytics YOLO train/val layout."
    )
    parser.add_argument(
        "--source",
        default="data/review_samples/20260518-221734-reviewed-good/data/cat_training",
        help="Reviewed source dataset root with images/<camera> and labels/<camera>",
    )
    parser.add_argument(
        "--output",
        default="data/fine_tune/cat_v1",
        help="Output YOLO dataset root",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--fixed-train-repeats",
        type=int,
        default=1,
        help=(
            "Total training copies for fixed-camera originals. "
            "Use sparingly; more real fixed frames are better."
        ),
    )
    parser.add_argument(
        "--augment-train",
        type=int,
        default=0,
        help="Augmented copies per training image. Requires albumentations and opencv.",
    )
    parser.add_argument(
        "--augment-fixed-extra",
        type=int,
        default=0,
        help="Additional augmented copies per fixed-camera training image.",
    )
    parser.add_argument("--fixed-camera", default="fixed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = prepare_finetune_dataset(
        source_root=Path(args.source),
        output_root=Path(args.output),
        val_fraction=args.val_fraction,
        fixed_train_repeats=args.fixed_train_repeats,
        augment_train_multiplier=args.augment_train,
        fixed_augment_multiplier=args.augment_fixed_extra,
        fixed_camera=args.fixed_camera,
        overwrite=args.overwrite,
    )

    print(f"prepared {summary.output_root}")
    print(
        f"train={summary.train_count} "
        f"val={summary.val_count} "
        f"augmented={summary.augmented_count}"
    )
    for (split, camera), count in sorted(summary.by_split_and_camera.items()):
        print(f"{split}/{camera}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
