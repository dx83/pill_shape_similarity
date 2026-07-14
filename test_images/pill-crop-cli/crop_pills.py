"""Standalone TorchScript pill-cropping command-line utility."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    import torchvision  # noqa: F401 - registers torchvision::nms used by the model
    from PIL import Image, ImageOps, UnidentifiedImageError
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Run setup.cmd before using this utility."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "crop_best.torchscript"


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    pad_left: int
    pad_top: int


@dataclass(frozen=True)
class CropModel:
    module: torch.jit.ScriptModule
    device: torch.device
    image_size: tuple[int, int]
    names: dict[int, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect pills in an image and save every detected crop.",
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input image path")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output image path")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="TorchScript model path (default: bundled model)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("ARGOS_DEVICE", "cpu"),
        help="PyTorch inference device (default: ARGOS_DEVICE or cpu)",
    )
    parser.add_argument(
        "--confidence",
        default=0.25,
        type=float,
        help="Minimum confidence from 0 to 1 (default: 0.25)",
    )
    return parser


def load_model(model_path: Path, device_name: str) -> CropModel:
    if not model_path.is_file():
        raise FileNotFoundError(f"TorchScript model not found: {model_path}")

    device = torch.device(device_name)
    extra_files: dict[str, Any] = {"config.txt": ""}
    module = torch.jit.load(
        str(model_path),
        map_location=device,
        _extra_files=extra_files,
    ).eval()
    metadata = parse_metadata(extra_files.get("config.txt"))
    image_size = metadata_image_size(metadata)
    raw_names = metadata.get("names", {"0": "pill"})
    if isinstance(raw_names, list):
        names = {index: str(label) for index, label in enumerate(raw_names)}
    else:
        names = {int(class_id): str(label) for class_id, label in raw_names.items()}
    return CropModel(module, device, image_size, names)


def detect(model: CropModel, image: Image.Image, confidence: float) -> list[Detection]:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    tensor, transform = letterbox_tensor(image, model.image_size, model.device)
    with torch.inference_mode():
        output = model.module(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[-1] < 6:
        raise RuntimeError("unexpected TorchScript crop-model output")

    width, height = image.size
    detections: list[Detection] = []
    for prediction in output[0].detach().cpu():
        score = float(prediction[4])
        if score < confidence:
            continue

        class_id = int(prediction[5])
        x1 = clamp((float(prediction[0]) - transform.pad_left) / transform.scale, width)
        y1 = clamp((float(prediction[1]) - transform.pad_top) / transform.scale, height)
        x2 = clamp((float(prediction[2]) - transform.pad_left) / transform.scale, width)
        y2 = clamp((float(prediction[3]) - transform.pad_top) / transform.scale, height)
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            Detection(
                x1,
                y1,
                x2,
                y2,
                score,
                class_id,
                model.names.get(class_id, str(class_id)),
            )
        )

    detections.sort(key=lambda detection: detection.confidence, reverse=True)
    return detections


def crop_pills(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    device: str,
    confidence: float,
) -> int:
    try:
        with Image.open(input_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
    except (FileNotFoundError, IsADirectoryError, PermissionError, UnidentifiedImageError, OSError) as exc:
        print(f"Unable to read input image '{input_path}': {exc}", file=sys.stderr)
        return 1

    try:
        model = load_model(model_path, device)
        detections = detect(model, image, confidence)
    except Exception as exc:
        print(f"Pill detection failed: {exc}", file=sys.stderr)
        return 1

    crops = []
    for detection in detections:
        bounds = crop_bounds(detection, *image.size)
        if bounds is not None:
            crops.append((detection, image.crop(bounds)))

    if not crops:
        print("No pill was detected in the input image.", file=sys.stderr)
        return 2

    output_paths = numbered_output_paths(output_path, len(crops))
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for (_, crop), crop_output_path in zip(crops, output_paths):
            crop.save(crop_output_path)
    except (PermissionError, OSError, ValueError) as exc:
        print(f"Unable to write pill crop: {exc}", file=sys.stderr)
        return 1

    for (detection, crop), crop_output_path in zip(crops, output_paths):
        print(
            f"Saved '{crop_output_path}' "
            f"({crop.width}x{crop.height}, confidence={detection.confidence:.4f})"
        )
    print(f"Saved {len(crops)} pill crop(s).")
    return 0


def parse_metadata(raw_metadata: Any) -> dict[str, Any]:
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    return json.loads(raw_metadata) if raw_metadata else {}


def metadata_image_size(metadata: dict[str, Any]) -> tuple[int, int]:
    image_size = metadata.get("imgsz", [640, 640])
    if isinstance(image_size, int):
        return image_size, image_size
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    raise ValueError("invalid imgsz in TorchScript model metadata")


def letterbox_tensor(
    image: Image.Image,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, LetterboxTransform]:
    image = image.convert("RGB")
    source_width, source_height = image.size
    target_height, target_width = image_size
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    horizontal_padding = (target_width - resized_width) / 2
    vertical_padding = (target_height - resized_height) / 2
    left = round(horizontal_padding - 0.1)
    top = round(vertical_padding - 0.1)
    canvas = Image.new("RGB", (target_width, target_height), (114, 114, 114))
    canvas.paste(resized, (left, top))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    return tensor, LetterboxTransform(scale, left, top)


def crop_bounds(detection: Detection, width: int, height: int) -> tuple[int, int, int, int] | None:
    left = max(0, min(floor(detection.x1), width))
    top = max(0, min(floor(detection.y1), height))
    right = max(0, min(ceil(detection.x2), width))
    bottom = max(0, min(ceil(detection.y2), height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def numbered_output_paths(output_path: Path, count: int) -> list[Path]:
    if count == 1:
        return [output_path]
    return [
        output_path.with_name(f"{output_path.stem}_{index}{output_path.suffix}")
        for index in range(1, count + 1)
    ]


def clamp(value: float, limit: int) -> float:
    return max(0.0, min(value, float(limit)))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return crop_pills(
        args.input,
        args.output,
        args.model,
        args.device,
        args.confidence,
    )


if __name__ == "__main__":
    raise SystemExit(main())
