"""Shared image segmentation and handcrafted pill-shape feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MASK_SIZE = 256
EFD_HARMONICS = 12
RADIAL_BINS = 64
RADIAL_FREQUENCIES = 16

BASIC_FEATURE_NAMES = (
    "log_aspect_ratio",
    "circularity",
    "eccentricity",
    "rectangularity",
    "solidity",
    "convexity",
    "concavity",
    "symmetry_best_axis",
    "symmetry_other_axis",
)
EFD_FEATURE_NAMES = tuple(
    name
    for harmonic in range(1, EFD_HARMONICS + 1)
    for name in (f"efd_{harmonic:02d}_major", f"efd_{harmonic:02d}_minor")
)
RADIAL_FEATURE_NAMES = tuple(
    f"radial_fft_{frequency:02d}" for frequency in range(1, RADIAL_FREQUENCIES + 1)
)
HU_FEATURE_NAMES = tuple(f"hu_{index}" for index in range(1, 8))
FEATURE_NAMES = BASIC_FEATURE_NAMES + EFD_FEATURE_NAMES + RADIAL_FEATURE_NAMES + HU_FEATURE_NAMES
FEATURE_GROUPS = {
    "basic": (0, len(BASIC_FEATURE_NAMES)),
    "efd": (len(BASIC_FEATURE_NAMES), len(BASIC_FEATURE_NAMES) + len(EFD_FEATURE_NAMES)),
    "radial": (
        len(BASIC_FEATURE_NAMES) + len(EFD_FEATURE_NAMES),
        len(BASIC_FEATURE_NAMES) + len(EFD_FEATURE_NAMES) + len(RADIAL_FEATURE_NAMES),
    ),
    "hu": (len(FEATURE_NAMES) - len(HU_FEATURE_NAMES), len(FEATURE_NAMES)),
}
DEFAULT_GROUP_WEIGHTS = {"basic": 0.40, "efd": 0.35, "radial": 0.20, "hu": 0.05}


class ShapeExtractionError(RuntimeError):
    """Raised when a usable pill silhouette cannot be extracted."""


@dataclass(frozen=True)
class MaskCandidate:
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]


def read_image(path: Path) -> np.ndarray:
    """Read an image through imdecode so Unicode Windows paths are supported."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise ShapeExtractionError(f"Cannot read image: {path}") from error
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ShapeExtractionError(f"Unsupported or corrupt image: {path}")
    return image


def _fill_and_smooth(mask: np.ndarray) -> np.ndarray:
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    short_side = min(mask.shape[:2])
    open_size = max(1, int(round(short_side * 0.003)))
    close_size = max(3, int(round(short_side * 0.012)))
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size | 1, open_size | 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size | 1, close_size | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _foreground_from_background(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    border = max(2, int(round(min(height, width) * 0.025)))
    border_pixels = np.concatenate(
        (
            lab[:border].reshape(-1, 3),
            lab[-border:].reshape(-1, 3),
            lab[:, :border].reshape(-1, 3),
            lab[:, -border:].reshape(-1, 3),
        )
    )
    background = np.median(border_pixels, axis=0)
    distances = np.linalg.norm(lab - background, axis=2)
    border_distances = np.linalg.norm(border_pixels - background, axis=1)
    # Product source images often place dark rulers and labels on the border.
    # The 95th percentile tolerates that contamination while retaining pale pills.
    threshold = max(12.0, float(np.percentile(border_distances, 95.0) + 7.0))
    return _fill_and_smooth(distances > threshold)


def _component_candidates(mask: np.ndarray, max_objects: int) -> list[MaskCandidate]:
    height, width = mask.shape
    image_area = height * width
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    raw: list[MaskCandidate] = []
    for label in range(1, component_count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if area < max(64, int(image_area * 0.0015)):
            continue
        if box_width < width * 0.025 or box_height < height * 0.025:
            continue
        fill_ratio = area / max(box_width * box_height, 1)
        if fill_ratio < 0.22:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        component = _fill_and_smooth(component)
        actual_area = int(np.count_nonzero(component))
        raw.append(MaskCandidate(component, actual_area, (x, y, box_width, box_height)))
    if not raw:
        raise ShapeExtractionError("No pill-like foreground component was found")
    raw.sort(key=lambda candidate: candidate.area, reverse=True)
    relative_minimum = raw[0].area * 0.15
    return [candidate for candidate in raw if candidate.area >= relative_minimum][:max_objects]


def extract_masks(image: np.ndarray, max_objects: int = 4) -> list[MaskCandidate]:
    """Extract one or more pill silhouettes from transparent or opaque input."""
    if max_objects < 1:
        raise ValueError("max_objects must be at least 1")
    alpha: np.ndarray | None = None
    if image.ndim == 3 and image.shape[2] == 4:
        candidate_alpha = image[:, :, 3]
        if np.any(candidate_alpha < 250) and np.any(candidate_alpha > 5):
            alpha = candidate_alpha
    mask = _fill_and_smooth(alpha > 8) if alpha is not None else _foreground_from_background(image)
    return _component_candidates(mask, max_objects=max_objects)


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ShapeExtractionError("The silhouette has no contour")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 8 or cv2.contourArea(contour) < 32:
        raise ShapeExtractionError("The silhouette contour is too small")
    return contour[:, 0, :].astype(np.float64)


def _canonical_mask(mask: np.ndarray, contour: np.ndarray) -> np.ndarray:
    points = contour.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    center = mean[0]
    major = eigenvectors[0]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), 1.0)
    rotated = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rotated > 0)
    if not len(xs):
        raise ShapeExtractionError("The silhouette disappeared during normalization")
    crop = rotated[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    padding = 12
    target = MASK_SIZE - 2 * padding
    scale = min(target / crop.shape[1], target / crop.shape[0])
    resized = cv2.resize(
        crop,
        (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
    y = (MASK_SIZE - resized.shape[0]) // 2
    x = (MASK_SIZE - resized.shape[1]) // 2
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return output


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = first > 0
    second_bool = second > 0
    union = np.count_nonzero(first_bool | second_bool)
    return float(np.count_nonzero(first_bool & second_bool) / max(union, 1))


def _efd_singular_values(contour: np.ndarray, harmonics: int) -> np.ndarray:
    closed = np.vstack((contour, contour[0]))
    delta = np.diff(closed, axis=0)
    dt = np.linalg.norm(delta, axis=1)
    valid = dt > 1e-8
    delta = delta[valid]
    dt = dt[valid]
    if len(dt) < 4:
        raise ShapeExtractionError("Not enough distinct contour points for EFD")
    t_end = np.cumsum(dt)
    total = float(t_end[-1])
    t_start = np.concatenate(([0.0], t_end[:-1]))
    derivative = delta / dt[:, None]
    descriptors: list[float] = []
    matrices: list[np.ndarray] = []
    for harmonic in range(1, harmonics + 1):
        omega = 2.0 * np.pi * harmonic / total
        coefficient = total / (2.0 * np.pi**2 * harmonic**2)
        cos_delta = np.cos(omega * t_end) - np.cos(omega * t_start)
        sin_delta = np.sin(omega * t_end) - np.sin(omega * t_start)
        a, c = coefficient * np.sum(derivative * cos_delta[:, None], axis=0)
        b, d = coefficient * np.sum(derivative * sin_delta[:, None], axis=0)
        matrices.append(np.array([[a, b], [c, d]], dtype=np.float64))
    scale = max(float(np.linalg.svd(matrices[0], compute_uv=False)[0]), 1e-8)
    for matrix in matrices:
        singular_values = np.linalg.svd(matrix, compute_uv=False) / scale
        descriptors.extend(float(value) for value in singular_values)
    return np.asarray(descriptors, dtype=np.float64)


def _radial_fft(mask: np.ndarray, bins: int, frequencies: int) -> np.ndarray:
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        raise ShapeExtractionError("The normalized silhouette is empty")
    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    ys, xs = np.where(mask > 0)
    dx = xs - center_x
    dy = ys - center_y
    radii = np.hypot(dx, dy)
    angles = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    indices = np.floor(angles * bins / (2.0 * np.pi)).astype(int) % bins
    signature = np.zeros(bins, dtype=np.float64)
    np.maximum.at(signature, indices, radii)
    missing = signature == 0
    if np.any(missing):
        known = np.flatnonzero(~missing)
        if not len(known):
            raise ShapeExtractionError("Cannot create radial descriptor")
        extended_x = np.concatenate((known - bins, known, known + bins))
        extended_y = np.tile(signature[known], 3)
        signature[missing] = np.interp(np.flatnonzero(missing), extended_x, extended_y)
    signature /= max(float(np.mean(signature)), 1e-8)
    spectrum = np.abs(np.fft.rfft(signature)) / bins
    return spectrum[1 : frequencies + 1]


def _log_hu_moments(contour: np.ndarray) -> np.ndarray:
    moments = cv2.moments(contour.astype(np.float32))
    hu = cv2.HuMoments(moments).flatten()
    return -np.log10(np.abs(hu) + 1e-30)


def mask_to_features(mask: np.ndarray) -> np.ndarray:
    """Convert a binary silhouette into the fixed handcrafted feature vector."""
    contour = _largest_contour(mask)
    contour_cv = contour.astype(np.float32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(contour_cv))
    perimeter = float(cv2.arcLength(contour_cv, True))
    rectangle = cv2.minAreaRect(contour_cv)
    width, height = rectangle[1]
    short_side = min(width, height)
    long_side = max(width, height)
    if area <= 0 or perimeter <= 0 or short_side <= 0:
        raise ShapeExtractionError("The silhouette geometry is degenerate")
    hull = cv2.convexHull(contour_cv)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))
    circularity = float(np.clip(4.0 * np.pi * area / perimeter**2, 0.0, 1.0))
    rectangularity = float(np.clip(area / (width * height), 0.0, 1.0))
    solidity = float(np.clip(area / max(hull_area, 1e-8), 0.0, 1.0))
    convexity = float(np.clip(hull_perimeter / perimeter, 0.0, 1.0))
    if len(contour_cv) >= 5:
        _, axes, _ = cv2.fitEllipse(contour_cv)
        minor_axis, major_axis = sorted(axes)
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis / max(major_axis, 1e-8)) ** 2)))
    else:
        eccentricity = 0.0
    canonical = _canonical_mask(mask, contour)
    symmetry_values = sorted(
        (
            _mask_iou(canonical, np.fliplr(canonical)),
            _mask_iou(canonical, np.flipud(canonical)),
        ),
        reverse=True,
    )
    basic = np.asarray(
        (
            np.log(max(long_side / short_side, 1.0)),
            circularity,
            eccentricity,
            rectangularity,
            solidity,
            convexity,
            1.0 - solidity,
            symmetry_values[0],
            symmetry_values[1],
        ),
        dtype=np.float64,
    )
    vector = np.concatenate(
        (
            basic,
            _efd_singular_values(contour, EFD_HARMONICS),
            _radial_fft(canonical, RADIAL_BINS, RADIAL_FREQUENCIES),
            _log_hu_moments(contour),
        )
    )
    if vector.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise ShapeExtractionError("The feature vector contains invalid values")
    return vector.astype(np.float32)


def extract_features_from_path(path: Path, max_objects: int = 4) -> list[tuple[np.ndarray, MaskCandidate]]:
    image = read_image(path)
    candidates = extract_masks(image, max_objects=max_objects)
    return [(mask_to_features(candidate.mask), candidate) for candidate in candidates]


def robust_scaler(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(vectors, axis=0)
    q25, q75 = np.percentile(vectors, (25.0, 75.0), axis=0)
    scale = q75 - q25
    fallback = np.std(vectors, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return center.astype(np.float32), scale.astype(np.float32)


def feature_dimension_weights(
    group_weights: dict[str, float] | None = None,
) -> np.ndarray:
    weights = DEFAULT_GROUP_WEIGHTS if group_weights is None else group_weights
    output = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    for group, (start, end) in FEATURE_GROUPS.items():
        output[start:end] = np.sqrt(float(weights[group]) / (end - start))
    return output


def transform_features(
    vectors: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    dimension_weights: np.ndarray,
) -> np.ndarray:
    standardized = np.clip((vectors - center) / scale, -10.0, 10.0)
    return (standardized * dimension_weights).astype(np.float32)
