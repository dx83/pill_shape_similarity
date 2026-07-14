"""Predict pill shape classes with a handcrafted prototype index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shape_features import FEATURE_NAMES, extract_features_from_path, transform_features


ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "handcrafted_shape_index.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pill shape from silhouette similarity")
    parser.add_argument("query", type=Path, help="Transparent pill crop or an opaque source image")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def load_index(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Shape index not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
    if str(index.get("index_type")) != "handcrafted_shape_prototypes":
        raise ValueError("The file is not a handcrafted shape prototype index")
    index_feature_names = tuple(str(value) for value in index["feature_names"])
    if index_feature_names != FEATURE_NAMES:
        raise ValueError("The index feature definition does not match this program")
    prototypes = index["prototypes"]
    if prototypes.ndim != 2 or prototypes.shape[1] != len(FEATURE_NAMES):
        raise ValueError("The index prototype matrix has an invalid shape")
    if prototypes.shape[0] != len(index["shape_names"]):
        raise ValueError("The number of prototypes and class names is different")
    return index


def ranked_results(
    distances: np.ndarray,
    index: dict[str, np.ndarray],
    top_k: int,
    nearest_item_indices: np.ndarray,
    centroid_distances: np.ndarray,
) -> list[dict[str, object]]:
    shape_names = index["shape_names"]
    order = np.argsort(distances)[: min(top_k, len(shape_names))]
    temperature = max(float(np.median(distances)), 1e-6)
    relative_scores = np.exp(-distances / temperature)
    relative_scores /= max(float(np.sum(relative_scores)), 1e-12)
    results: list[dict[str, object]] = []
    for rank, class_index in enumerate(order, start=1):
        radius = float(index["class_radii"][class_index])
        nearest_index = int(nearest_item_indices[class_index])
        results.append(
            {
                "rank": rank,
                "shape": str(shape_names[class_index]),
                "distance": float(distances[class_index]),
                "class_centroid_distance": float(centroid_distances[class_index]),
                "relative_score": float(relative_scores[class_index]),
                "acceptance_radius": radius,
                "within_training_range": bool(distances[class_index] <= radius),
                "item_count": int(index["class_item_counts"][class_index]),
                "image_count": int(index["class_image_counts"][class_index]),
                "nearest_item_id": str(index["item_ids"][nearest_index]),
                "nearest_item_name": str(index["item_names"][nearest_index]),
                "nearest_item_distance": float(distances[class_index]),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be at least 1")
    if args.max_objects < 1:
        raise ValueError("max-objects must be at least 1")
    if not args.query.is_file():
        raise FileNotFoundError(f"Query image not found: {args.query}")

    index = load_index(args.index)
    extracted = extract_features_from_path(args.query, max_objects=args.max_objects)
    raw_vectors = np.stack([features for features, _ in extracted])
    vectors = transform_features(
        raw_vectors,
        index["scaler_center"],
        index["scaler_scale"],
        index["dimension_weights"],
    )
    prototypes = index["prototypes"].astype(np.float32)
    centroid_object_distances = np.linalg.norm(
        vectors[:, None, :] - prototypes[None, :, :], axis=2
    )
    consensus_centroid_distances = np.median(centroid_object_distances, axis=0)
    item_object_distances = np.linalg.norm(
        vectors[:, None, :] - index["item_vectors"][None, :, :], axis=2
    )
    class_count = len(index["shape_names"])
    object_distances = np.empty((len(vectors), class_count), dtype=np.float32)
    object_nearest_items = np.empty((len(vectors), class_count), dtype=np.int64)
    consensus_distances = np.empty(class_count, dtype=np.float32)
    consensus_nearest_items = np.empty(class_count, dtype=np.int64)
    for class_index, shape in enumerate(index["shape_names"]):
        item_indices = np.flatnonzero(index["item_shapes"] == shape)
        class_distances = item_object_distances[:, item_indices]
        nearest_positions = np.argmin(class_distances, axis=1)
        object_distances[:, class_index] = class_distances[
            np.arange(len(vectors)), nearest_positions
        ]
        object_nearest_items[:, class_index] = item_indices[nearest_positions]
        per_item_consensus = np.median(class_distances, axis=0)
        best_position = int(np.argmin(per_item_consensus))
        consensus_distances[class_index] = per_item_consensus[best_position]
        consensus_nearest_items[class_index] = item_indices[best_position]
    consensus = ranked_results(
        consensus_distances,
        index,
        args.top_k,
        consensus_nearest_items,
        consensus_centroid_distances,
    )
    objects = []
    for object_index, ((_, candidate), distances) in enumerate(zip(extracted, object_distances), start=1):
        objects.append(
            {
                "object": object_index,
                "bbox": list(candidate.bbox),
                "area": candidate.area,
                "results": ranked_results(
                    distances,
                    index,
                    args.top_k,
                    object_nearest_items[object_index - 1],
                    centroid_object_distances[object_index - 1],
                ),
            }
        )
    top_margin = None
    if len(consensus) >= 2:
        top_margin = float(consensus[1]["distance"] - consensus[0]["distance"])
    output = {
        "query": str(args.query.resolve()),
        "index": str(args.index.resolve()),
        "detected_objects": len(objects),
        "aggregation": "minimum same-class item-prototype distance after median aggregation across detected pill objects",
        "top_margin": top_margin,
        "consensus": consensus,
        "objects": objects,
        "score_note": "relative_score is a ranking aid, not a calibrated probability",
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    print(f"Query: {args.query.resolve()}")
    print(f"Detected pill objects: {len(objects)}")
    print("Consensus shape ranking")
    print("-" * 96)
    for result in consensus:
        range_text = "yes" if result["within_training_range"] else "no"
        print(
            f"{result['rank']:>2}. shape={result['shape']}  distance={result['distance']:.6f}  "
            f"relative_score={result['relative_score']:.4f}  within_range={range_text}  "
            f"items={result['item_count']} images={result['image_count']}"
        )
        if "nearest_item_id" in result:
            print(
                f"    nearest_item={result['nearest_item_id']}  "
                f"name={result['nearest_item_name']}  centroid_distance={result['class_centroid_distance']:.6f}"
            )
    if top_margin is not None:
        print(f"Top-1/Top-2 distance margin: {top_margin:.6f}")
    if len(objects) > 1:
        print("\nPer-object Top-1")
        for result in objects:
            first = result["results"][0]
            print(
                f"object={result['object']} bbox={tuple(result['bbox'])} "
                f"shape={first['shape']} distance={first['distance']:.6f}"
            )


if __name__ == "__main__":
    main()
