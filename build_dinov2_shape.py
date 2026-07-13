"""DINOv2로 의약품 제형별 대표 임베딩 인덱스를 생성한다."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = ROOT / "pill_images_original"
DEFAULT_CSV = ROOT / "csv" / "OpenData_pill_final_cleaned_updated.csv"
DEFAULT_WEIGHTS = ROOT / "dinov2_vits14_pretrain.pth"
DEFAULT_OUTPUT = ROOT / "dinov2_shape_index.pt"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ID_COLUMN = "품목일련번호"
SHAPE_COLUMN = "의약품제형"
ROTATION_ANGLES = (0, 90, 180, 270)


class ShapeImageDataset(Dataset):
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                views = [
                    self.transform(image.rotate(angle, expand=True))
                    for angle in ROTATION_ANGLES
                ]
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}") from error
        return torch.stack(views), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINOv2 의약품 제형별 대표 임베딩 인덱스를 생성합니다."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "xpu"), default="cpu"
    )
    return parser.parse_args()


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 있는 CUDA GPU가 없습니다.")
    if requested == "xpu" and not is_xpu_available():
        raise RuntimeError(
            "XPU를 요청했지만 사용할 수 있는 Intel XPU가 없습니다. "
            "XPU 지원 PyTorch와 Intel 그래픽 드라이버를 확인하세요."
        )
    return torch.device(requested)


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"지원하지 않는 체크포인트 형식입니다: {path}")
    return state_dict


def load_model(weights: Path, device: torch.device) -> torch.nn.Module:
    print("DINOv2 ViT-S/14 모델 구조를 불러오는 중...")
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=False,
        trust_repo=True,
    )
    model.load_state_dict(load_checkpoint(weights), strict=True)
    return model.eval().to(device)


def load_shape_by_id(csv_path: Path) -> tuple[dict[str, str], int]:
    shapes_by_id: dict[str, set[str]] = defaultdict(set)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {ID_COLUMN, SHAPE_COLUMN}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV에 품목일련번호, 의약품제형 컬럼이 필요합니다.")
        for row in reader:
            item_id = (row[ID_COLUMN] or "").strip()
            shape = (row[SHAPE_COLUMN] or "").strip()
            if item_id and shape:
                shapes_by_id[item_id].add(shape)

    shape_by_id: dict[str, str] = {}
    conflict_count = 0
    for item_id, shapes in shapes_by_id.items():
        if len(shapes) == 1:
            shape_by_id[item_id] = next(iter(shapes))
        else:
            conflict_count += 1
    return shape_by_id, conflict_count


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size는 1 이상, workers는 0 이상이어야 합니다.")
    if not args.images.is_dir():
        raise NotADirectoryError(f"이미지 폴더가 없습니다: {args.images}")
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {args.csv}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"가중치 파일이 없습니다: {args.weights}")

    all_image_paths = sorted(
        path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_image_paths:
        raise RuntimeError("처리할 이미지가 없습니다.")

    shape_by_id, conflict_count = load_shape_by_id(args.csv)
    image_paths: list[Path] = []
    image_item_ids: list[str] = []
    image_shapes: list[str] = []
    skipped_without_shape = 0
    for path in all_image_paths:
        item_id = path.stem.split("_", maxsplit=1)[0].strip()
        shape = shape_by_id.get(item_id)
        if not shape:
            skipped_without_shape += 1
            continue
        image_paths.append(path)
        image_item_ids.append(item_id)
        image_shapes.append(shape)

    if not image_paths:
        raise RuntimeError("CSV의 제형 정보와 매칭되는 이미지가 없습니다.")

    device = select_device(args.device)
    model = load_model(args.weights, device)
    loader = DataLoader(
        ShapeImageDataset(image_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    sums_by_shape: dict[str, torch.Tensor] = {}
    image_count_by_shape: dict[str, int] = defaultdict(int)
    item_ids_by_shape: dict[str, set[str]] = defaultdict(set)
    processed = 0
    print(
        f"{len(image_paths):,}개 이미지로 제형 대표 벡터 생성 시작 "
        f"(device={device}, rotations={ROTATION_ANGLES})"
    )
    with torch.inference_mode():
        for image_views, indices in loader:
            batch_size, view_count = image_views.shape[:2]
            embedding_sum: torch.Tensor | None = None
            for view_number in range(view_count):
                images = image_views[:, view_number].to(
                    device, non_blocking=device.type == "cuda"
                )
                view_embeddings = F.normalize(
                    model(images).float(), p=2, dim=1
                )
                if embedding_sum is None:
                    embedding_sum = view_embeddings
                else:
                    embedding_sum += view_embeddings
            if embedding_sum is None:
                raise RuntimeError("회전 증강 이미지가 생성되지 않았습니다.")
            embeddings = F.normalize(
                embedding_sum / view_count, p=2, dim=1
            ).cpu()
            for embedding, index in zip(embeddings, indices.tolist()):
                shape = image_shapes[index]
                if shape in sums_by_shape:
                    sums_by_shape[shape] += embedding
                else:
                    sums_by_shape[shape] = embedding.clone()
                image_count_by_shape[shape] += 1
                item_ids_by_shape[shape].add(image_item_ids[index])
            processed += batch_size
            print(f"\r처리: {processed:,}/{len(image_paths):,}", end="", flush=True)
    print()

    shape_names = sorted(sums_by_shape)
    shape_embeddings = torch.stack(
        [sums_by_shape[shape] / image_count_by_shape[shape] for shape in shape_names]
    )
    shape_embeddings = F.normalize(shape_embeddings, p=2, dim=1)
    classes = [
        {
            "shape": shape,
            "image_count": image_count_by_shape[shape],
            "item_count": len(item_ids_by_shape[shape]),
        }
        for shape in shape_names
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "index_type": "dinov2_shape_centroids",
            "model_name": "dinov2_vits14",
            "embedding_dim": int(shape_embeddings.shape[1]),
            "augmentation": {
                "type": "fixed_rotations",
                "angles": list(ROTATION_ANGLES),
                "view_aggregation": "mean_then_l2_normalize",
            },
            "shape_embeddings": shape_embeddings,
            "classes": classes,
        },
        args.output,
    )
    print(f"제형 클래스 수: {len(classes):,}")
    print(f"제형 미매칭으로 제외된 이미지: {skipped_without_shape:,}")
    print(f"제형 충돌로 제외된 품목 ID: {conflict_count:,}")
    print(f"제형 인덱스 저장 완료: {args.output.resolve()}")


if __name__ == "__main__":
    main()
