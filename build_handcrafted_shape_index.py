"""Build robust pill-shape class prototypes from handcrafted silhouette features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from shape_features import (
    DEFAULT_GROUP_WEIGHTS,
    FEATURE_NAMES,
    IMAGE_EXTENSIONS,
    extract_features_from_path,
    feature_dimension_weights,
    robust_scaler,
    transform_features,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = ROOT / "pill_images_original"
DEFAULT_CSV = ROOT / "csv" / "OpenData_pill_final_cleaned_updated.csv"
DEFAULT_OUTPUT = ROOT / "handcrafted_shape_index.npz"
DEFAULT_REPORT = ROOT / "handcrafted_shape_build_report.json"
ID_COLUMN = "\ud488\ubaa9\uc77c\ub828\ubc88\ud638"
NAME_COLUMN = "\ud488\ubaa9\uba85"
SHAPE_COLUMN = "\uc758\uc57d\ud488\uc81c\ud615"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a handcrafted pill-shape prototype index")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None, help="Process only N matched images for a smoke test")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def load_item_metadata(
    csv_path: Path,
) -> tuple[dict[str, tuple[str, str]], set[str], set[str], int]:
    rows_by_id: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {ID_COLUMN, NAME_COLUMN, SHAPE_COLUMN}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or ())
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        for row in reader:
            item_id = (row[ID_COLUMN] or "").strip()
            item_name = (row[NAME_COLUMN] or "").strip()
            shape = (row[SHAPE_COLUMN] or "").strip()
            if item_id:
                rows_by_id[item_id].append((item_name, shape))

    metadata: dict[str, tuple[str, str]] = {}
    conflicts: set[str] = set()
    missing_shapes: set[str] = set()
    duplicate_rows = 0
    for item_id, rows in rows_by_id.items():
        duplicate_rows += max(0, len(rows) - 1)
        shapes = {shape for _, shape in rows if shape}
        if len(shapes) > 1:
            conflicts.add(item_id)
            continue
        if not shapes:
            missing_shapes.add(item_id)
            continue
        name = next((name for name, _ in rows if name), "")
        metadata[item_id] = (name, next(iter(shapes)))
    return metadata, conflicts, missing_shapes, duplicate_rows


def image_item_id(path: Path) -> str:
    return path.stem.split("_", maxsplit=1)[0].strip()


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be at least 1")
    if args.progress_every < 1:
        raise ValueError("progress-every must be at least 1")
    if not args.images.is_dir():
        raise NotADirectoryError(f"Image directory not found: {args.images}")
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    metadata, conflict_ids, missing_shape_ids, duplicate_rows = load_item_metadata(args.csv)
    all_images = sorted(
        path for path in args.images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    matched: list[tuple[Path, str, str, str]] = []
    skipped_missing_metadata = 0
    skipped_conflict_images = 0
    for path in all_images:
        item_id = image_item_id(path)
        if item_id in conflict_ids:
            skipped_conflict_images += 1
            continue
        item = metadata.get(item_id)
        if item is None:
            skipped_missing_metadata += 1
            continue
        item_name, shape = item
        matched.append((path, item_id, item_name, shape))
    if args.limit is not None:
        matched = matched[: args.limit]
    if not matched:
        raise RuntimeError("No non-conflicting CSV rows matched an image")

    print(f"Matched images: {len(matched):,}; conflicting item IDs excluded: {len(conflict_ids):,}")
    vectors_by_item: dict[str, list[np.ndarray]] = defaultdict(list)
    item_name_by_id: dict[str, str] = {}
    shape_by_id: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    for number, (path, item_id, item_name, shape) in enumerate(matched, start=1):
        try:
            extracted = extract_features_from_path(path, max_objects=1)
            vectors_by_item[item_id].append(extracted[0][0])
            item_name_by_id[item_id] = item_name
            shape_by_id[item_id] = shape
        except Exception as error:
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
        if number % args.progress_every == 0 or number == len(matched):
            print(f"\rFeature extraction: {number:,}/{len(matched):,}; failures={len(failures):,}", end="", flush=True)
    print()
    if not vectors_by_item:
        raise RuntimeError("Feature extraction failed for every image")

    item_ids = sorted(vectors_by_item)
    raw_item_vectors = np.stack([np.median(vectors_by_item[item_id], axis=0) for item_id in item_ids]).astype(np.float32)
    scaler_center, scaler_scale = robust_scaler(raw_item_vectors)
    dimension_weights = feature_dimension_weights()
    item_vectors = transform_features(raw_item_vectors, scaler_center, scaler_scale, dimension_weights)

    item_indices_by_shape: dict[str, list[int]] = defaultdict(list)
    for index, item_id in enumerate(item_ids):
        item_indices_by_shape[shape_by_id[item_id]].append(index)
    shape_names = sorted(item_indices_by_shape)
    prototypes = np.stack(
        [np.median(item_vectors[item_indices_by_shape[shape]], axis=0) for shape in shape_names]
    ).astype(np.float32)

    per_shape_distances: dict[str, np.ndarray] = {}
    all_training_distances: list[float] = []
    for shape, prototype in zip(shape_names, prototypes):
        distances = np.linalg.norm(item_vectors[item_indices_by_shape[shape]] - prototype, axis=1)
        per_shape_distances[shape] = distances
        all_training_distances.extend(float(value) for value in distances)
    global_radius = float(np.percentile(all_training_distances, 95.0))
    class_radii = np.asarray(
        [
            np.percentile(per_shape_distances[shape], 95.0)
            if len(per_shape_distances[shape]) >= 5
            else global_radius
            for shape in shape_names
        ],
        dtype=np.float32,
    )
    class_image_counts = Counter(
        shape_by_id[item_id] for item_id, values in vectors_by_item.items() for _ in values
    )
    class_item_counts = np.asarray([len(item_indices_by_shape[shape]) for shape in shape_names], dtype=np.int32)
    class_image_count_array = np.asarray([class_image_counts[shape] for shape in shape_names], dtype=np.int32)
    item_names = np.asarray([item_name_by_id[item_id] for item_id in item_ids])
    item_shapes = np.asarray([shape_by_id[item_id] for item_id in item_ids])
    item_image_counts = np.asarray([len(vectors_by_item[item_id]) for item_id in item_ids], dtype=np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format_version=np.asarray(1, dtype=np.int32),
        index_type=np.asarray("handcrafted_shape_prototypes"),
        feature_names=np.asarray(FEATURE_NAMES),
        shape_names=np.asarray(shape_names),
        prototypes=prototypes,
        scaler_center=scaler_center,
        scaler_scale=scaler_scale,
        dimension_weights=dimension_weights,
        class_radii=class_radii,
        class_item_counts=class_item_counts,
        class_image_counts=class_image_count_array,
        item_ids=np.asarray(item_ids),
        item_names=item_names,
        item_shapes=item_shapes,
        item_vectors=item_vectors,
        item_image_counts=item_image_counts,
        group_names=np.asarray(tuple(DEFAULT_GROUP_WEIGHTS)),
        group_weights=np.asarray(tuple(DEFAULT_GROUP_WEIGHTS.values()), dtype=np.float32),
    )

    report = {
        "index": str(args.output.resolve()),
        "source_csv": str(args.csv.resolve()),
        "source_images": str(args.images.resolve()),
        "all_image_files": len(all_images),
        "matched_images_considered": len(matched),
        "successful_images": int(sum(len(values) for values in vectors_by_item.values())),
        "failed_images": len(failures),
        "indexed_items": len(item_ids),
        "indexed_shapes": len(shape_names),
        "duplicate_csv_rows": duplicate_rows,
        "conflicting_item_ids_excluded": sorted(conflict_ids),
        "item_ids_without_shape": sorted(missing_shape_ids),
        "images_excluded_for_conflict": skipped_conflict_images,
        "images_without_usable_metadata": skipped_missing_metadata,
        "classes": [
            {
                "shape": shape,
                "item_count": int(class_item_counts[index]),
                "image_count": int(class_image_count_array[index]),
                "acceptance_radius": float(class_radii[index]),
            }
            for index, shape in enumerate(shape_names)
        ],
        "failures": failures,
    }
    save_report(args.report, report)
    print(f"Shapes: {len(shape_names):,}; items: {len(item_ids):,}; successful images: {report['successful_images']:,}")
    print(f"Index saved: {args.output.resolve()}")
    print(f"Build report saved: {args.report.resolve()}")


if __name__ == "__main__":
    main()
