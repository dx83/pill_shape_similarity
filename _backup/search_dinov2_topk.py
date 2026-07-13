"""DINOv2 인덱스에서 입력 이미지와 유사한 알약 이미지 Top-K를 검색한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "dinov2_vits14_pretrain.pth"
DEFAULT_INDEX = ROOT / "dinov2_pill_index.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 코사인 유사도 Top-K 검색")
    parser.add_argument("query", type=Path, help="검색할 이미지 경로")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "xpu"), default="cpu"
    )
    parser.add_argument(
        "--unique-item",
        action="store_true",
        help="같은 품목일련번호는 최고 점수 이미지 하나만 출력",
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
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=False,
        trust_repo=True,
    )
    model.load_state_dict(load_checkpoint(weights), strict=True)
    return model.eval().to(device)


def extract_query_embedding(
    model: torch.nn.Module, query_path: Path, device: torch.device
) -> torch.Tensor:
    transform = transforms.Compose(
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
    with Image.open(query_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        return F.normalize(model(tensor).float(), p=2, dim=1).cpu()[0]


def safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top-k는 1 이상이어야 합니다.")
    for path, description in (
        (args.query, "검색 이미지"),
        (args.index, "인덱스"),
        (args.weights, "가중치"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} 파일이 없습니다: {path}")

    index = safe_torch_load(args.index)
    if index.get("model_name") != "dinov2_vits14":
        raise ValueError("인덱스가 dinov2_vits14 모델로 생성되지 않았습니다.")
    embeddings = index["embeddings"].float()
    records = index["records"]
    if len(embeddings) != len(records):
        raise ValueError("인덱스의 임베딩 수와 레코드 수가 다릅니다.")

    device = select_device(args.device)
    model = load_model(args.weights, device)
    query = extract_query_embedding(model, args.query, device)
    similarities = embeddings @ query  # L2 정규화 벡터의 내적 = 코사인 유사도
    order = torch.argsort(similarities, descending=True).tolist()

    query_resolved = args.query.resolve()
    results: list[tuple[float, dict]] = []
    seen_item_ids: set[str] = set()
    for index_number in order:
        record = records[index_number]
        if Path(record["path"]).resolve() == query_resolved:
            continue
        if args.unique_item and record["item_id"] in seen_item_ids:
            continue
        results.append((float(similarities[index_number]), record))
        seen_item_ids.add(record["item_id"])
        if len(results) == args.top_k:
            break

    print(f"검색 이미지: {query_resolved}")
    print(f"Top-{len(results)} 결과")
    print("-" * 100)
    for rank, (score, record) in enumerate(results, start=1):
        print(
            f"{rank:>2}. similarity={score:.6f}  "
            f"item_id={record['item_id']}  shape={record['shape']}  "
            f"side={record['side']}"
        )
        print(f"    품목명: {record['product_name']}")
        print(f"    이미지: {record['path']}")


if __name__ == "__main__":
    main()
