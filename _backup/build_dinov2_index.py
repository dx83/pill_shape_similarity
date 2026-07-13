"""DINOv2 ViT-S/14로 알약 이미지의 검색용 임베딩 인덱스를 생성한다.

CSV와 이미지 원본은 읽기만 하며 수정하지 않는다. DINOv2 공식 소스가 PyTorch
Hub 캐시에 없다면 첫 실행 시 facebookresearch/dinov2에서 소스 코드를 받는다.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUTPUT = ROOT / "dinov2_pill_index.pt"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ID_COLUMN = "품목일련번호"
NAME_COLUMN = "품목명"
SHAPE_COLUMN = "의약품제형"


class PillImageDataset(Dataset):
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
                tensor = self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}") from error
        return tensor, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 알약 이미지 인덱스를 생성합니다.")
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
    except TypeError:  # PyTorch 2.0 이전 버전 호환
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


def load_metadata(csv_path: Path) -> dict[str, dict[str, str]]:
    values_by_id: dict[str, dict[str, set[str]]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {ID_COLUMN, NAME_COLUMN, SHAPE_COLUMN}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV에 품목일련번호, 품목명, 의약품제형 컬럼이 필요합니다.")
        for row in reader:
            item_id = (row[ID_COLUMN] or "").strip()
            if not item_id:
                continue
            values = values_by_id.setdefault(item_id, {"names": set(), "shapes": set()})
            values["names"].add((row[NAME_COLUMN] or "").strip())
            values["shapes"].add((row[SHAPE_COLUMN] or "").strip())

    metadata: dict[str, dict[str, str]] = {}
    for item_id, values in values_by_id.items():
        metadata[item_id] = {
            "product_name": (
                next(iter(values["names"]))
                if len(values["names"]) == 1
                else "(CSV 품목명 중복)"
            ),
            "shape": (
                next(iter(values["shapes"]))
                if len(values["shapes"]) == 1
                else "(CSV 제형 충돌)"
            ),
        }
    return metadata


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

    image_paths = sorted(
        path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError("인덱싱할 이미지가 없습니다.")

    device = select_device(args.device)
    metadata_by_id = load_metadata(args.csv)
    model = load_model(args.weights, device)
    loader = DataLoader(
        PillImageDataset(image_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    embedding_batches: list[torch.Tensor] = []
    processed = 0
    print(f"{len(image_paths):,}개 이미지 임베딩 추출 시작 (device={device})")
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            embeddings = F.normalize(model(images).float(), p=2, dim=1)
            embedding_batches.append(embeddings.cpu())
            processed += len(images)
            print(f"\r처리: {processed:,}/{len(image_paths):,}", end="", flush=True)
    print()

    item_ids = [path.stem.split("_", maxsplit=1)[0].strip() for path in image_paths]
    records = []
    for path, item_id in zip(image_paths, item_ids):
        info = metadata_by_id.get(item_id, {})
        records.append(
            {
                "path": str(path.resolve()),
                "item_id": item_id,
                "product_name": info.get("product_name", "(CSV 매칭 없음)"),
                "shape": info.get("shape", "(CSV 매칭 없음)"),
                "side": path.parent.name.lower(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": "dinov2_vits14",
            "embedding_dim": 384,
            "embeddings": torch.cat(embedding_batches, dim=0),
            "records": records,
        },
        args.output,
    )
    print(f"인덱스 저장 완료: {args.output.resolve()}")


if __name__ == "__main__":
    main()
