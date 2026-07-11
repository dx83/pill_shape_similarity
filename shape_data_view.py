"""CSV의 의약품제형별 실제 이미지 파일 수를 조회한다.

이 스크립트는 CSV와 이미지 디렉터리를 읽기만 하며 원본을 변경하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "csv" / "OpenData_pill_final_cleaned_updated.csv"
DEFAULT_IMAGE_DIR = BASE_DIR / "pill_images_original"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
ID_COLUMN = "품목일련번호"
SHAPE_COLUMN = "의약품제형"
EMPTY_SHAPE = "(제형 없음)"
UNKNOWN_SHAPE = "(CSV 매칭 없음)"
CONFLICT_SHAPE = "(CSV 제형 충돌)"


def load_shapes(csv_path: Path) -> tuple[dict[str, str], int, int]:
    """품목일련번호와 제형의 대응 관계를 반환한다."""
    shapes_by_id: dict[str, str] = {}
    duplicate_rows = 0
    conflict_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV에 헤더가 없습니다.")

        missing_columns = {
            column for column in (ID_COLUMN, SHAPE_COLUMN) if column not in reader.fieldnames
        }
        if missing_columns:
            raise ValueError(
                "CSV에 필요한 컬럼이 없습니다: " + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            item_id = (row[ID_COLUMN] or "").strip()
            shape = (row[SHAPE_COLUMN] or "").strip() or EMPTY_SHAPE
            if not item_id:
                continue

            previous_shape = shapes_by_id.get(item_id)
            if previous_shape is not None:
                duplicate_rows += 1
                if previous_shape != shape:
                    shapes_by_id[item_id] = CONFLICT_SHAPE
                    conflict_ids.add(item_id)
            else:
                shapes_by_id[item_id] = shape

    return shapes_by_id, duplicate_rows, len(conflict_ids)


def count_images(
    image_dir: Path, shapes_by_id: dict[str, str]
) -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
    """제형별 전체/앞/뒤 이미지 수와 고유 품목 번호를 집계한다."""
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"front": 0, "back": 0, "other": 0, "total": 0}
    )
    item_ids: dict[str, set[str]] = defaultdict(set)

    for image_path in image_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        item_id = image_path.stem.split("_", maxsplit=1)[0].strip()
        shape = shapes_by_id.get(item_id, UNKNOWN_SHAPE)
        side = image_path.parent.name.lower()
        side_key = side if side in {"front", "back"} else "other"

        counts[shape][side_key] += 1
        counts[shape]["total"] += 1
        item_ids[shape].add(item_id)

    return counts, item_ids


def print_report(
    counts: dict[str, dict[str, int]],
    item_ids: dict[str, set[str]],
    duplicate_rows: int,
    conflict_count: int,
) -> None:
    """집계 결과를 이미지 수가 많은 제형부터 출력한다."""
    rows = sorted(counts, key=lambda shape: (-counts[shape]["total"], shape))
    shape_width = max([len("의약품제형"), *(len(shape) for shape in rows)])

    print(
        f"{'의약품제형':<{shape_width}}  {'고유 품목':>9}  "
        f"{'앞 이미지':>9}  {'뒤 이미지':>9}  {'기타':>7}  {'전체 이미지':>11}"
    )
    print("-" * (shape_width + 58))

    for shape in rows:
        values = counts[shape]
        print(
            f"{shape:<{shape_width}}  {len(item_ids[shape]):>9,}  "
            f"{values['front']:>9,}  {values['back']:>9,}  "
            f"{values['other']:>7,}  {values['total']:>11,}"
        )

    print("-" * (shape_width + 58))
    print(
        f"합계{' ' * max(shape_width - 2, 0)}  "
        f"{len(set().union(*item_ids.values())):>9,}  "
        f"{sum(value['front'] for value in counts.values()):>9,}  "
        f"{sum(value['back'] for value in counts.values()):>9,}  "
        f"{sum(value['other'] for value in counts.values()):>7,}  "
        f"{sum(value['total'] for value in counts.values()):>11,}"
    )

    if duplicate_rows:
        print(f"\n참고: CSV의 중복 품목 행 {duplicate_rows:,}개는 한 품목으로 처리했습니다.")
    if conflict_count:
        print(
            f"주의: 중복 행의 제형이 서로 다른 품목 {conflict_count:,}개는 "
            f"{CONFLICT_SHAPE}로 분리했습니다."
        )
    if UNKNOWN_SHAPE in counts:
        print(
            f"주의: CSV에서 품목일련번호를 찾지 못한 이미지가 "
            f"{counts[UNKNOWN_SHAPE]['total']:,}개 있습니다."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV를 기준으로 의약품제형별 실제 이미지 수를 조회합니다."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV 파일 경로")
    parser.add_argument(
        "--images", type=Path, default=DEFAULT_IMAGE_DIR, help="이미지 최상위 폴더 경로"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {args.csv}")
    if not args.images.is_dir():
        raise NotADirectoryError(f"이미지 폴더를 찾을 수 없습니다: {args.images}")

    shapes_by_id, duplicate_rows, conflict_count = load_shapes(args.csv)
    counts, item_ids = count_images(args.images, shapes_by_id)
    print_report(counts, item_ids, duplicate_rows, conflict_count)


if __name__ == "__main__":
    main()
