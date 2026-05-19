from __future__ import annotations

import csv
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class YoloSourceSample:
    camera: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class FinetuneDatasetSummary:
    output_root: Path
    train_count: int
    val_count: int
    augmented_count: int
    by_split_and_camera: dict[tuple[str, str], int]


def prepare_finetune_dataset(
    *,
    source_root: str | Path,
    output_root: str | Path,
    val_fraction: float = 0.2,
    fixed_train_repeats: int = 1,
    augment_train_multiplier: int = 0,
    fixed_augment_multiplier: int = 0,
    fixed_camera: str = "fixed",
    class_id: int = 0,
    class_name: str = "cat",
    overwrite: bool = False,
    seed: int | None = None,
) -> FinetuneDatasetSummary:
    del seed  # Reserved for future shuffled splits; current split is chronological by filename.
    source = Path(source_root)
    output = Path(output_root)
    samples = discover_yolo_samples(source)
    train_samples, val_samples = split_samples_by_camera(samples, val_fraction=val_fraction)

    if fixed_train_repeats < 1:
        raise ValueError("fixed_train_repeats must be at least 1")
    if augment_train_multiplier < 0 or fixed_augment_multiplier < 0:
        raise ValueError("augmentation multipliers must be non-negative")

    _prepare_output_root(output, overwrite=overwrite)
    _write_dataset_yaml(output, class_id=class_id, class_name=class_name)
    (output / "classes.txt").write_text(f"{class_name}\n", encoding="utf-8")

    manifest_rows: list[dict[str, str]] = []
    counts: Counter[tuple[str, str]] = Counter()
    augmented_count = 0

    for sample in train_samples:
        repeat_count = fixed_train_repeats if sample.camera == fixed_camera else 1
        for repeat_index in range(1, repeat_count + 1):
            suffix = "" if repeat_index == 1 else f"_repeat{repeat_index:02d}"
            _copy_sample(
                sample=sample,
                output_root=output,
                split="train",
                output_stem=f"{sample.image_path.stem}{suffix}",
                manifest_rows=manifest_rows,
                augmented=False,
            )
            counts[("train", sample.camera)] += 1

        sample_augments = augment_train_multiplier
        if sample.camera == fixed_camera:
            sample_augments += fixed_augment_multiplier
        if sample_augments:
            augmenter = _build_augmenter(horizontal_flip=sample.camera != fixed_camera)
            cv2 = _import_cv2()
            for aug_index in range(1, sample_augments + 1):
                did_write = _augment_sample(
                    sample=sample,
                    output_root=output,
                    split="train",
                    output_stem=f"{sample.image_path.stem}_aug{aug_index:02d}",
                    augmenter=augmenter,
                    cv2=cv2,
                    manifest_rows=manifest_rows,
                )
                if did_write:
                    counts[("train", sample.camera)] += 1
                    augmented_count += 1

    for sample in val_samples:
        _copy_sample(
            sample=sample,
            output_root=output,
            split="val",
            output_stem=sample.image_path.stem,
            manifest_rows=manifest_rows,
            augmented=False,
        )
        counts[("val", sample.camera)] += 1

    _write_manifest(output / "manifest.csv", manifest_rows)
    return FinetuneDatasetSummary(
        output_root=output,
        train_count=sum(count for (split, _), count in counts.items() if split == "train"),
        val_count=sum(count for (split, _), count in counts.items() if split == "val"),
        augmented_count=augmented_count,
        by_split_and_camera=dict(counts),
    )


def discover_yolo_samples(source_root: str | Path) -> list[YoloSourceSample]:
    root = Path(source_root)
    image_root = root / "images"
    label_root = root / "labels"
    if not image_root.exists():
        raise ValueError(f"missing image root: {image_root}")
    if not label_root.exists():
        raise ValueError(f"missing label root: {label_root}")

    samples: list[YoloSourceSample] = []
    image_paths = (
        path for path in image_root.glob("*/*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in sorted(image_paths):
        camera = image_path.parent.name
        label_path = label_root / camera / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise ValueError(f"missing label for {image_path}: {label_path}")
        samples.append(
            YoloSourceSample(camera=camera, image_path=image_path, label_path=label_path)
        )

    if not samples:
        raise ValueError(f"no YOLO image/label pairs found under {root}")
    return samples


def split_samples_by_camera(
    samples: list[YoloSourceSample],
    *,
    val_fraction: float,
) -> tuple[list[YoloSourceSample], list[YoloSourceSample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be >= 0.0 and < 1.0")

    grouped: dict[str, list[YoloSourceSample]] = defaultdict(list)
    for sample in sorted(samples, key=lambda item: (item.camera, item.image_path.name)):
        grouped[sample.camera].append(sample)

    train: list[YoloSourceSample] = []
    val: list[YoloSourceSample] = []
    for camera_samples in grouped.values():
        val_count = _validation_count(len(camera_samples), val_fraction)
        if val_count:
            train.extend(camera_samples[:-val_count])
            val.extend(camera_samples[-val_count:])
        else:
            train.extend(camera_samples)

    return train, val


def _validation_count(sample_count: int, val_fraction: float) -> int:
    if sample_count <= 1 or val_fraction == 0.0:
        return 0
    requested = round(sample_count * val_fraction)
    return max(1, min(sample_count - 1, int(requested)))


def _prepare_output_root(output: Path, *, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise ValueError(f"output already exists; pass overwrite=True to replace it: {output}")
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_dataset_yaml(output: Path, *, class_id: int, class_name: str) -> None:
    (output / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  {class_id}: {class_name}\n",
        encoding="utf-8",
    )


def _copy_sample(
    *,
    sample: YoloSourceSample,
    output_root: Path,
    split: str,
    output_stem: str,
    manifest_rows: list[dict[str, str]],
    augmented: bool,
) -> None:
    image_path = output_root / "images" / split / f"{output_stem}{sample.image_path.suffix.lower()}"
    label_path = output_root / "labels" / split / f"{output_stem}.txt"
    shutil.copy2(sample.image_path, image_path)
    shutil.copy2(sample.label_path, label_path)
    manifest_rows.append(
        _manifest_row(
            split=split,
            camera=sample.camera,
            image_path=image_path,
            label_path=label_path,
            source_image=sample.image_path,
            source_label=sample.label_path,
            augmented=augmented,
            output_root=output_root,
        )
    )


def _augment_sample(
    *,
    sample: YoloSourceSample,
    output_root: Path,
    split: str,
    output_stem: str,
    augmenter: Any,
    cv2: Any,
    manifest_rows: list[dict[str, str]],
) -> bool:
    image = cv2.imread(str(sample.image_path))
    if image is None:
        raise ValueError(f"failed to read image for augmentation: {sample.image_path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, labels = _read_yolo_labels(sample.label_path)
    transformed = augmenter(image=rgb_image, bboxes=bboxes, class_labels=labels)
    transformed_bboxes = transformed["bboxes"]
    transformed_labels = transformed["class_labels"]
    if bboxes and not transformed_bboxes:
        return False

    image_path = output_root / "images" / split / f"{output_stem}{sample.image_path.suffix.lower()}"
    label_path = output_root / "labels" / split / f"{output_stem}.txt"
    bgr_image = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), bgr_image):
        raise RuntimeError(f"failed to write augmented image: {image_path}")
    _write_yolo_labels(label_path, transformed_bboxes, transformed_labels)
    manifest_rows.append(
        _manifest_row(
            split=split,
            camera=sample.camera,
            image_path=image_path,
            label_path=label_path,
            source_image=sample.image_path,
            source_label=sample.label_path,
            augmented=True,
            output_root=output_root,
        )
    )
    return True


def _build_augmenter(*, horizontal_flip: bool) -> Any:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "Albumentations is required for offline augmentation. "
            "Install the training extra or run with augmentation multipliers set to 0."
        ) from exc

    transforms: list[Any] = []
    if horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.25))
    transforms.extend(
        [
            A.Affine(
                translate_percent=(-0.03, 0.03),
                scale=(0.95, 1.05),
                rotate=(-5, 5),
                border_mode=0,
                fill=0,
                p=0.35,
            ),
            A.RandomBrightnessContrast(p=0.45),
            A.HueSaturationValue(p=0.2),
            A.OneOf([A.GaussianBlur(blur_limit=3), A.MotionBlur(blur_limit=3)], p=0.2),
            A.ISONoise(p=0.2),
            _coarse_dropout(A),
        ]
    )
    return A.Compose(transforms, bbox_params=_bbox_params(A))


def _bbox_params(A: Any) -> Any:
    try:
        return A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
            clip=True,
            filter_invalid_bboxes=True,
        )
    except TypeError:
        return A.BboxParams(
            coord_format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
            clip_bboxes_on_input=True,
            filter_invalid_bboxes=True,
        )


def _coarse_dropout(A: Any) -> Any:
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.02, 0.06),
            hole_width_range=(0.02, 0.06),
            fill="random_uniform",
            p=0.15,
        )
    except TypeError:
        return A.CoarseDropout(
            max_holes=3,
            max_height=24,
            max_width=24,
            fill_value=0,
            p=0.15,
        )


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for offline augmentation. "
            "Install the bench/training dependencies or run without augmentation."
        ) from exc
    return cv2


def _read_yolo_labels(label_path: Path) -> tuple[list[list[float]], list[int]]:
    bboxes: list[list[float]] = []
    labels: list[int] = []
    label_lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(label_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        labels.append(int(parts[0]))
        bboxes.append([float(value) for value in parts[1:]])
    return bboxes, labels


def _write_yolo_labels(label_path: Path, bboxes: list[list[float]], labels: list[int]) -> None:
    lines = [
        f"{int(label)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
        for bbox, label in zip(bboxes, labels, strict=True)
    ]
    label_path.write_text("".join(lines), encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["split", "camera", "image", "label", "source_image", "source_label", "augmented"]
    with path.open("w", encoding="utf-8", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(
    *,
    split: str,
    camera: str,
    image_path: Path,
    label_path: Path,
    source_image: Path,
    source_label: Path,
    augmented: bool,
    output_root: Path,
) -> dict[str, str]:
    return {
        "split": split,
        "camera": camera,
        "image": image_path.relative_to(output_root).as_posix(),
        "label": label_path.relative_to(output_root).as_posix(),
        "source_image": source_image.as_posix(),
        "source_label": source_label.as_posix(),
        "augmented": "true" if augmented else "false",
    }
