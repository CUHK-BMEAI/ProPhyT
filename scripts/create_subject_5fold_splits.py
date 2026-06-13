#!/usr/bin/env python3
"""Create reproducible subject-level 5-fold JSON splits for SAM-Med2D.

The generated fold directories contain the JSON files expected by
DataLoader.TrainingDataset and DataLoader.TestingDataset:

  image2label_train.json
  label2image_valid.json
  label2image_test.json

For cross-validation, the held-out fold is written to both valid and test.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


MASK_SUFFIX_RE = re.compile(r"_(core|penumbra)_000$")


def subject_from_image(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def slice_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(f"{row}\n")


def ensure_dir_symlink(link_path: Path, target_path: Path) -> None:
    target_path = target_path.resolve()
    if link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"{link_path} exists and is not a symlink")
    link_path.symlink_to(target_path, target_is_directory=True)


def build_image_records(data_root: Path):
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks dir: {masks_dir}")

    records = []
    missing = []
    for image_path in sorted(images_dir.glob("*.png"), key=lambda p: (subject_from_image(p), slice_index(p), p.name)):
        core_mask = masks_dir / f"{image_path.stem}_core_000.png"
        penumbra_mask = masks_dir / f"{image_path.stem}_penumbra_000.png"
        if not core_mask.exists() or not penumbra_mask.exists():
            missing.append((image_path, core_mask.exists(), penumbra_mask.exists()))
            continue
        records.append(
            {
                "subject": subject_from_image(image_path),
                "slice": slice_index(image_path),
                "image": image_path.as_posix(),
                "masks": [core_mask.as_posix(), penumbra_mask.as_posix()],
            }
        )

    if missing:
        preview = "\n".join(f"  {p} core={has_core} penumbra={has_penumbra}" for p, has_core, has_penumbra in missing[:10])
        raise RuntimeError(f"Found images without both masks ({len(missing)} total):\n{preview}")
    if not records:
        raise RuntimeError(f"No image records found under {data_root}")
    return records


def assign_subject_folds(subject_to_records, n_folds: int, seed: int):
    rng = random.Random(seed)
    subjects = list(subject_to_records)
    subject_tiebreak = {subject: rng.random() for subject in subjects}
    subjects.sort(key=lambda s: (-len(subject_to_records[s]), subject_tiebreak[s], s))

    fold_subjects = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for subject in subjects:
        fold_idx = min(range(n_folds), key=lambda i: (fold_sizes[i], len(fold_subjects[i]), i))
        fold_subjects[fold_idx].append(subject)
        fold_sizes[fold_idx] += len(subject_to_records[subject])

    return fold_subjects, fold_sizes


def mappings_for_records(records):
    ordered = sorted(records, key=lambda r: (r["subject"], r["slice"], r["image"]))
    image2label = {r["image"]: r["masks"] for r in ordered}

    label2image = {}
    for r in ordered:
        for mask_path in r["masks"]:
            label2image[mask_path] = r["image"]
    return image2label, label2image


def validate_fold_manifest(manifest, subject_to_records) -> None:
    all_subjects = set(subject_to_records)
    subject_to_fold = manifest["subject_to_fold"]

    if set(subject_to_fold) != all_subjects:
        missing = sorted(all_subjects - set(subject_to_fold))
        extra = sorted(set(subject_to_fold) - all_subjects)
        raise RuntimeError(f"Subject-to-fold mismatch. missing={missing[:5]} extra={extra[:5]}")

    heldout_seen = set()
    for fold in manifest["folds"]:
        fold_idx = fold["fold"]
        heldout_subjects = set(fold["valid_test_subject_ids"])
        train_subjects = all_subjects - heldout_subjects

        if heldout_seen & heldout_subjects:
            dup = sorted(heldout_seen & heldout_subjects)
            raise RuntimeError(f"Subjects appear as held-out in multiple folds: {dup[:5]}")
        heldout_seen.update(heldout_subjects)

        for subject in heldout_subjects:
            if subject_to_fold[subject] != fold_idx:
                raise RuntimeError(f"Subject {subject} has inconsistent fold assignment")

        train_images = sum(len(subject_to_records[subject]) for subject in train_subjects)
        heldout_images = sum(len(subject_to_records[subject]) for subject in heldout_subjects)

        if train_subjects & heldout_subjects:
            overlap = sorted(train_subjects & heldout_subjects)
            raise RuntimeError(f"Fold {fold_idx} train/held-out overlap: {overlap[:5]}")
        if fold["train_subjects"] != len(train_subjects) or fold["valid_test_subjects"] != len(heldout_subjects):
            raise RuntimeError(f"Fold {fold_idx} subject counts are inconsistent")
        if fold["train_images"] != train_images or fold["valid_test_images"] != heldout_images:
            raise RuntimeError(f"Fold {fold_idx} image counts are inconsistent")
        if fold["train_masks"] != train_images * 2 or fold["valid_test_masks"] != heldout_images * 2:
            raise RuntimeError(f"Fold {fold_idx} mask counts are inconsistent")

    if heldout_seen != all_subjects:
        missing = sorted(all_subjects - heldout_seen)
        raise RuntimeError(f"Subjects never held out: {missing[:5]}")


def main():
    parser = argparse.ArgumentParser(description="Create subject-level K-fold splits for SAM-Med2D JSON loaders.")
    parser.add_argument("--data-root", default="data_penumbra_noblank_withvalid")
    parser.add_argument("--output-root", default="data_penumbra_noblank_withvalid_5fold_subject_seed42")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output root.")
    parser.add_argument(
        "--link-data-dirs",
        action="store_true",
        help="Create images/ and masks/ symlinks inside each fold directory for loaders that expect data_root/images.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    if output_root.exists() and not args.force:
        raise FileExistsError(f"{output_root} already exists. Use --force to overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)

    records = build_image_records(data_root)
    subject_to_records = defaultdict(list)
    for record in records:
        subject_to_records[record["subject"]].append(record)

    fold_subjects, fold_sizes = assign_subject_folds(subject_to_records, args.n_folds, args.seed)
    all_subjects = set(subject_to_records)

    manifest = {
        "data_root": data_root.as_posix(),
        "output_root": output_root.as_posix(),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_subjects": len(subject_to_records),
        "n_images": len(records),
        "n_masks": len(records) * 2,
        "subject_to_fold": {},
        "folds": [],
    }

    for fold_idx, heldout_subjects in enumerate(fold_subjects):
        heldout_subject_set = set(heldout_subjects)
        train_subjects = sorted(all_subjects - heldout_subject_set)

        train_records = [r for subject in train_subjects for r in subject_to_records[subject]]
        heldout_records = [r for subject in sorted(heldout_subject_set) for r in subject_to_records[subject]]

        train_i2l, train_l2i = mappings_for_records(train_records)
        heldout_i2l, heldout_l2i = mappings_for_records(heldout_records)

        fold_dir = output_root / f"fold{fold_idx}"
        write_json(fold_dir / "image2label_train.json", train_i2l)
        write_json(fold_dir / "label2image_train.json", train_l2i)
        write_json(fold_dir / "image2label_valid.json", heldout_i2l)
        write_json(fold_dir / "label2image_valid.json", heldout_l2i)
        write_json(fold_dir / "image2label_test.json", heldout_i2l)
        write_json(fold_dir / "label2image_test.json", heldout_l2i)
        write_lines(fold_dir / "subjects_train.txt", train_subjects)
        write_lines(fold_dir / "subjects_valid_test.txt", sorted(heldout_subject_set))
        if args.link_data_dirs:
            ensure_dir_symlink(fold_dir / "images", data_root / "images")
            ensure_dir_symlink(fold_dir / "masks", data_root / "masks")

        for subject in heldout_subject_set:
            manifest["subject_to_fold"][subject] = fold_idx

        manifest["folds"].append(
            {
                "fold": fold_idx,
                "train_subjects": len(train_subjects),
                "valid_test_subjects": len(heldout_subject_set),
                "train_images": len(train_records),
                "valid_test_images": len(heldout_records),
                "train_masks": len(train_l2i),
                "valid_test_masks": len(heldout_l2i),
                "valid_test_subject_ids": sorted(heldout_subject_set),
            }
        )

    write_json(output_root / "manifest.json", manifest)
    validate_fold_manifest(manifest, subject_to_records)

    print(f"Wrote subject-level {args.n_folds}-fold splits to {output_root}")
    print("Validation passed: no subject overlap; every subject is held out exactly once.")
    print(f"Subjects: {len(subject_to_records)} | images: {len(records)} | masks: {len(records) * 2}")
    for fold in manifest["folds"]:
        print(
            f"fold{fold['fold']}: "
            f"train_subjects={fold['train_subjects']} train_images={fold['train_images']} | "
            f"valid/test_subjects={fold['valid_test_subjects']} valid/test_images={fold['valid_test_images']}"
        )


if __name__ == "__main__":
    main()
