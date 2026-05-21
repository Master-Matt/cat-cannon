from pathlib import Path

import pytest

from cat_cannon.app.finetune_dataset import prepare_finetune_dataset


def _write_sample(
    root: Path,
    camera: str,
    stem: str,
    label_line: str = "0 0.500000 0.500000 0.100000 0.100000\n",
) -> None:
    image_dir = root / "images" / camera
    label_dir = root / "labels" / camera
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.jpg").write_bytes(f"image-{stem}".encode())
    (label_dir / f"{stem}.txt").write_text(label_line, encoding="utf-8")


def test_prepare_finetune_dataset_splits_pairs_by_camera(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for index in range(4):
        _write_sample(source, "fixed", f"fixed_{index}")
    for index in range(5):
        _write_sample(source, "turret", f"turret_{index}")

    summary = prepare_finetune_dataset(
        source_root=source,
        output_root=tmp_path / "dataset",
        val_fraction=0.25,
        seed=7,
    )

    assert summary.train_count == 7
    assert summary.val_count == 2
    assert summary.by_split_and_camera == {
        ("train", "fixed"): 3,
        ("train", "turret"): 4,
        ("val", "fixed"): 1,
        ("val", "turret"): 1,
    }
    assert (tmp_path / "dataset" / "dataset.yaml").read_text(encoding="utf-8") == (
        f"path: {(tmp_path / 'dataset').resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: cat\n"
    )
    assert len(list((tmp_path / "dataset" / "images" / "train").glob("*.jpg"))) == 7
    assert len(list((tmp_path / "dataset" / "labels" / "train").glob("*.txt"))) == 7
    assert len(list((tmp_path / "dataset" / "images" / "val").glob("*.jpg"))) == 2
    assert len(list((tmp_path / "dataset" / "labels" / "val").glob("*.txt"))) == 2
    manifest = (tmp_path / "dataset" / "manifest.csv").read_text(encoding="utf-8")
    assert "split,camera,image,label,source_image,source_label,augmented\n" in manifest
    assert ",fixed," in manifest
    assert ",turret," in manifest


def test_prepare_finetune_dataset_can_repeat_fixed_training_samples(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for index in range(4):
        _write_sample(source, "fixed", f"fixed_{index}")
    for index in range(4):
        _write_sample(source, "turret", f"turret_{index}")

    summary = prepare_finetune_dataset(
        source_root=source,
        output_root=tmp_path / "dataset",
        val_fraction=0.25,
        seed=11,
        fixed_train_repeats=3,
    )

    assert summary.by_split_and_camera[("train", "fixed")] == 9
    assert summary.by_split_and_camera[("train", "turret")] == 3
    assert len(list((tmp_path / "dataset" / "images" / "train").glob("fixed_*.jpg"))) == 9
    assert len(list((tmp_path / "dataset" / "labels" / "train").glob("fixed_*.txt"))) == 9
    assert (tmp_path / "dataset" / "images" / "train" / "fixed_0_repeat02.jpg").exists()


def test_prepare_finetune_dataset_can_shuffle_split_with_seed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for index in range(10):
        _write_sample(source, "fixed", f"fixed_{index}")

    prepare_finetune_dataset(
        source_root=source,
        output_root=tmp_path / "dataset",
        val_fraction=0.2,
        seed=123,
    )

    manifest = (tmp_path / "dataset" / "manifest.csv").read_text(encoding="utf-8")
    assert "val,fixed,images/val/fixed_4.jpg" in manifest
    assert "val,fixed,images/val/fixed_0.jpg" in manifest


def test_prepare_finetune_dataset_preserves_source_class_names(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "classes.txt").write_text("cat\nperson\n", encoding="utf-8")
    _write_sample(source, "fixed", "fixed_cat")
    _write_sample(
        source,
        "turret",
        "turret_person",
        "1 0.250000 0.250000 0.200000 0.300000\n",
    )

    prepare_finetune_dataset(
        source_root=source,
        output_root=tmp_path / "dataset",
        val_fraction=0.0,
    )

    assert (tmp_path / "dataset" / "classes.txt").read_text(encoding="utf-8") == (
        "cat\nperson\n"
    )
    assert (tmp_path / "dataset" / "dataset.yaml").read_text(encoding="utf-8") == (
        f"path: {(tmp_path / 'dataset').resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: cat\n"
        "  1: person\n"
    )
    assert (tmp_path / "dataset" / "labels" / "train" / "turret_person.txt").read_text(
        encoding="utf-8"
    ) == "1 0.250000 0.250000 0.200000 0.300000\n"


def test_prepare_finetune_dataset_rejects_images_without_labels(tmp_path: Path) -> None:
    image_dir = tmp_path / "source" / "images" / "fixed"
    image_dir.mkdir(parents=True)
    (image_dir / "fixed_missing.jpg").write_bytes(b"image")

    with pytest.raises(ValueError, match="missing label"):
        prepare_finetune_dataset(
            source_root=tmp_path / "source",
            output_root=tmp_path / "dataset",
        )
