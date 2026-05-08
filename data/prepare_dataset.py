#!/usr/bin/env python3

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps, ImageEnhance


Labels = Dict[str, int]
Dataset = Dict[str, Labels]

AUGMENT_TRANSFORMS = [
    ("aug_hflip",    lambda img: ImageOps.mirror(img)),
    ("aug_vflip",    lambda img: ImageOps.flip(img)),
    ("aug_rot90",    lambda img: img.rotate(90, expand=True)),
    ("aug_rot270",   lambda img: img.rotate(270, expand=True)),
    ("aug_bright",   lambda img: ImageEnhance.Brightness(img).enhance(1.4)),
    ("aug_dark",     lambda img: ImageEnhance.Brightness(img).enhance(0.6)),
    ("aug_contrast", lambda img: ImageEnhance.Contrast(img).enhance(1.5)),
    ("aug_sat",      lambda img: ImageEnhance.Color(img).enhance(1.5)),
    ("aug_desat",    lambda img: ImageEnhance.Color(img).enhance(0.5)),
    ("aug_sharp",    lambda img: ImageEnhance.Sharpness(img).enhance(2.0)),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_dataset(raw: dict) -> Tuple[Dataset, List[str]]:
    if not isinstance(raw, dict):
        raise TypeError("JSON must be a dictionary.")

    classes = raw.get("classes")

    if classes:
        images_key = next(
            (key for key, value in raw.items() if key != "classes" and isinstance(value, dict)),
            None,
        )
        if images_key is None:
            raise ValueError("JSON has classes, but no image labels dictionary.")
        images = raw[images_key]
    else:
        images = raw

    if not images:
        return {}, classes or []

    first_value = next(iter(images.values()))

    if isinstance(first_value, dict):
        data = {
            str(filename): {str(label): int(value) for label, value in labels.items()}
            for filename, labels in images.items()
        }
        columns = list(next(iter(data.values())).keys())
        return data, columns

    if isinstance(first_value, list):
        if not classes:
            raise ValueError("Labels are lists, but classes are missing.")

        data = {}
        for filename, values in images.items():
            if len(values) != len(classes):
                raise ValueError(f"{filename}: labels count does not match classes count.")

            data[str(filename)] = {
                str(label): int(value)
                for label, value in zip(classes, values)
            }

        return data, [str(label) for label in classes]

    raise TypeError(f"Unsupported label format: {type(first_value)}")


def get_label_columns(data: Dataset) -> List[str]:
    columns = []
    seen = set()

    for labels in data.values():
        for label in labels:
            if label not in seen:
                columns.append(label)
                seen.add(label)

    return columns


def is_dress_only(labels: Labels) -> bool:
    active = [label for label, value in labels.items() if int(value) == 1]
    return active == ["dress"]


def remove_dress_only(data: Dataset) -> Dataset:
    cleaned = {
        filename: labels
        for filename, labels in data.items()
        if not is_dress_only(labels)
    }

    print(f"[Clean] Removed dress-only rows: {len(data) - len(cleaned)}")
    return cleaned


def drop_labels(data: Dataset, labels_to_drop: List[str]) -> Dataset:
    if not labels_to_drop:
        return data

    available = set(get_label_columns(data))
    valid = [label for label in labels_to_drop if label in available]

    if not valid:
        print("[Drop] No valid labels to drop.")
        return data

    print(f"[Drop] Removed label columns: {valid}")

    return {
        filename: {
            label: value
            for label, value in labels.items()
            if label not in valid
        }
        for filename, labels in data.items()
    }


def split_dataset(data: Dataset, val_size: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0 < val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")

    filenames = list(data.keys())
    random.Random(seed).shuffle(filenames)

    val_count = round(len(filenames) * val_size)
    val_files = sorted(filenames[:val_count])
    train_files = sorted(filenames[val_count:])

    print(f"[Split] Train: {len(train_files)}")
    print(f"[Split] Val:   {len(val_files)}")

    return train_files, val_files


def write_csv(path: Path, filenames: List[str], data: Dataset, columns: List[str], split_name: str = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["filename"]
    if split_name is not None:
        header.append("split")
    header.extend(columns)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filename in filenames:
            row = [filename]
            if split_name is not None:
                row.append(split_name)

            row.extend(int(data[filename].get(label, 0)) for label in columns)
            writer.writerow(row)

    print(f"[CSV] Saved: {path}")


def write_full_csv(path: Path, train_files: List[str], val_files: List[str], data: Dataset, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split"] + columns)

        for filename in train_files:
            row = [filename, "train"]
            row.extend(int(data[filename].get(label, 0)) for label in columns)
            writer.writerow(row)

        for filename in val_files:
            row = [filename, "val"]
            row.extend(int(data[filename].get(label, 0)) for label in columns)
            writer.writerow(row)

    print(f"[CSV] Saved: {path}")


def find_image(image_dir: Path, filename: str) -> Path:
    candidates = [
        image_dir / filename,
        image_dir / Path(filename).name,
        Path(filename),
    ]

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    return image_dir / filename


def resize_image(src: Path, dst: Path, size: int, mode: str) -> bool:
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")

        if mode == "pad":
            img = ImageOps.pad(img, (size, size), method=Image.Resampling.LANCZOS, color=(255, 255, 255))
        else:
            img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)

        save_kwargs = {}
        if dst.suffix.lower() in [".jpg", ".jpeg"]:
            save_kwargs = {"quality": 95, "optimize": True}

        img.save(dst, **save_kwargs)

    return True


def create_resized_folder(data: Dataset, image_dir: Path, output_dir: Path, size: int, mode: str) -> None:
    target_dir = output_dir / f"train_{size}"
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    missing = []

    for filename in data:
        src = find_image(image_dir, filename)
        dst = target_dir / filename

        try:
            if resize_image(src, dst, size, mode):
                saved += 1
            else:
                missing.append(filename)
        except Exception as error:
            missing.append(filename)
            print(f"[Resize {size}] Failed: {filename} -> {error}")

    print(f"[Resize {size}] Saved: {saved}/{len(data)} images")

    if missing:
        missing_path = output_dir / f"missing_images_{size}.txt"
        missing_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"[Resize {size}] Missing list saved: {missing_path}")


def clean_output(output_dir: Path) -> None:
    for name in ["train_384", "train_512"]:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)

    for name in [
        "train.csv", "val.csv", "dataset.csv",
        "missing_images_384.txt", "missing_images_512.txt",
        "augmentation_report.txt",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()


def count_label_frequencies(data: Dataset, columns: List[str]) -> Dict[str, int]:
    return {
        label: sum(1 for labels in data.values() if labels.get(label, 0) == 1)
        for label in columns
    }


def find_rare_labels(freq: Dict[str, int], total: int, threshold: float) -> List[str]:
    cutoff = total * threshold
    return [label for label, count in freq.items() if count < cutoff]


def build_augmented_filename(original: str, suffix: str) -> str:
    p = Path(original)
    return str(p.with_stem(f"{p.stem}_{suffix}"))


def _is_augmented(filename: str) -> bool:
    stem = Path(filename).stem
    return any(stem.endswith(f"_{suffix}") for suffix, _ in AUGMENT_TRANSFORMS)


def augment_rare_labels(
    data: Dataset,
    image_dir: Path,
    rare_labels: List[str],
    transforms: List[Tuple[str, callable]],
    max_augments_per_image: int,
    seed: int,
) -> Dataset:
    if not rare_labels:
        print("[Augment] No rare labels found — skipping augmentation.")
        return data

    rare_set = set(rare_labels)
    rng = random.Random(seed)

    candidates = [
        filename
        for filename, labels in data.items()
        if any(labels.get(lbl, 0) == 1 for lbl in rare_set) and not _is_augmented(filename)
    ]

    print(f"[Augment] Rare labels : {rare_labels}")
    print(f"[Augment] Images to augment: {len(candidates)}")

    selected_transforms = (
        rng.sample(transforms, max_augments_per_image)
        if len(transforms) > max_augments_per_image
        else transforms[:max_augments_per_image]
    )

    augmented: Dataset = {}
    saved = 0
    failed = 0

    for filename in candidates:
        src = find_image(image_dir, filename)

        if not src.exists():
            print(f"[Augment] Source not found, skipping: {filename}")
            failed += 1
            continue

        try:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")

                for suffix, transform_fn in selected_transforms:
                    aug_filename = build_augmented_filename(filename, suffix)
                    dst = image_dir / Path(aug_filename).name

                    if not dst.exists():
                        aug_img = transform_fn(img)
                        save_kwargs = {}
                        if dst.suffix.lower() in [".jpg", ".jpeg"]:
                            save_kwargs = {"quality": 95, "optimize": True}
                        aug_img.save(dst, **save_kwargs)

                    augmented[aug_filename] = dict(data[filename])
                    saved += 1

        except Exception as err:
            print(f"[Augment] Failed: {filename} -> {err}")
            failed += 1

    print(f"[Augment] Created : {saved} augmented images")
    if failed:
        print(f"[Augment] Failed  : {failed} source images")

    merged = dict(data)
    merged.update(augmented)
    return merged


def write_augmentation_report(
    path: Path,
    freq_before: Dict[str, int],
    freq_after: Dict[str, int],
    total_before: int,
    total_after: int,
    rare_labels: List[str],
    threshold: float,
) -> None:
    lines = [
        "Augmentation Report",
        "===================",
        f"Rarity threshold : {threshold:.0%} of dataset  ({round(total_before * threshold)} images)",
        f"Rare labels      : {rare_labels or 'none'}",
        f"Dataset size     : {total_before} -> {total_after} images",
        "",
        f"{'Label':<30} {'Before':>8} {'After':>8} {'Ratio':>8}",
        "-" * 58,
    ]

    for label in sorted(set(list(freq_before) + list(freq_after))):
        before = freq_before.get(label, 0)
        after = freq_after.get(label, 0)
        ratio = f"x{after / before:.1f}" if before else "N/A"
        marker = " *" if label in rare_labels else ""
        lines.append(f"{label:<30} {before:>8} {after:>8} {ratio:>8}{marker}")

    lines += ["", "* = rare label (was augmented)"]
    report = "\n".join(lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"[Augment] Report saved: {path}")
    print()
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="checkpoint_static.json")
    parser.add_argument("--imgs", default="train")
    parser.add_argument("--out", default="output")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-labels", nargs="+", default=[])
    parser.add_argument("--keep-dress-only", action="store_true")
    parser.add_argument("--resize-mode", choices=["pad", "crop"], default="pad")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument(
        "--aug-rare-threshold",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Labels present in fewer than FRAC*N images are augmented. 0 disables. (default: 0.1)",
    )
    parser.add_argument(
        "--aug-max-transforms",
        type=int,
        default=4,
        metavar="N",
        help="Max augmented variants per source image, randomly sampled. (default: 4)",
    )
    parser.add_argument("--aug-skip", action="store_true", help="Disable rare-label augmentation.")

    args = parser.parse_args()

    json_path = Path(args.json)
    image_dir = Path(args.imgs)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        clean_output(output_dir)

    raw = load_json(json_path)
    data, _ = extract_dataset(raw)

    print(f"[Load] Rows: {len(data)}")

    if not args.keep_dress_only:
        data = remove_dress_only(data)

    data = drop_labels(data, args.drop_labels)

    if not data:
        raise SystemExit("No data left after cleaning.")

    columns = get_label_columns(data)

    if not args.aug_skip and args.aug_rare_threshold > 0:
        freq_before = count_label_frequencies(data, columns)
        total_before = len(data)

        rare_labels = find_rare_labels(freq_before, total_before, args.aug_rare_threshold)

        data = augment_rare_labels(
            data=data,
            image_dir=image_dir,
            rare_labels=rare_labels,
            transforms=AUGMENT_TRANSFORMS,
            max_augments_per_image=args.aug_max_transforms,
            seed=args.seed,
        )

        freq_after = count_label_frequencies(data, columns)
        write_augmentation_report(
            path=output_dir / "augmentation_report.txt",
            freq_before=freq_before,
            freq_after=freq_after,
            total_before=total_before,
            total_after=len(data),
            rare_labels=rare_labels,
            threshold=args.aug_rare_threshold,
        )
    else:
        print("[Augment] Skipped.")

    train_files, val_files = split_dataset(data, args.val_size, args.seed)

    write_csv(output_dir / "train.csv", train_files, data, columns)
    write_csv(output_dir / "val.csv", val_files, data, columns)
    write_full_csv(output_dir / "dataset.csv", train_files, val_files, data, columns)

    create_resized_folder(data, image_dir, output_dir, 384, args.resize_mode)
    create_resized_folder(data, image_dir, output_dir, 512, args.resize_mode)

    print("[Done] Dataset prepared successfully.")
    print(f"[Done] Output folder: {output_dir}")


if __name__ == "__main__":
    main()