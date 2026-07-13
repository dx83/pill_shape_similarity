# `build_dinov2_index.py` 분석

## 1. 개요

이 스크립트는 알약 원본 이미지를 DINOv2 ViT-S/14 모델로 처리하여 검색용 임베딩 인덱스(`.pt`)를 생성한다.

이미지마다 384차원 특징 벡터를 추출하고 L2 정규화한 뒤, CSV에서 읽은 품목명과 의약품 제형 등의 메타데이터를 결합해 하나의 PyTorch 파일로 저장한다. CSV와 이미지 원본은 읽기만 하며 수정하지 않는다.

## 2. 기본 입출력

모든 기본 경로는 스크립트가 위치한 디렉터리를 기준으로 한다.

| 구분 | 기본값 | 용도 |
|---|---|---|
| 이미지 폴더 | `pill_images_original/` | 인덱싱할 알약 이미지 검색 |
| CSV | `csv/OpenData_pill_final_cleaned_updated.csv` | 품목 메타데이터 조회 |
| 모델 가중치 | `dinov2_vits14_pretrain.pth` | DINOv2 ViT-S/14 사전 학습 가중치 |
| 출력 파일 | `dinov2_pill_index.pt` | 임베딩과 레코드 저장 |

지원 이미지 확장자는 `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, `.webp`이다. 이미지 폴더의 모든 하위 디렉터리를 재귀적으로 탐색한다.

## 3. 명령행 옵션

| 옵션 | 형식/선택값 | 기본값 | 설명 |
|---|---|---|---|
| `--images` | 경로 | `pill_images_original` | 이미지 루트 폴더 |
| `--csv` | 경로 | `csv/OpenData_pill_final_cleaned_updated.csv` | 메타데이터 CSV |
| `--weights` | 경로 | `dinov2_vits14_pretrain.pth` | 모델 가중치 파일 |
| `--output` | 경로 | `dinov2_pill_index.pt` | 결과 저장 경로 |
| `--batch-size` | 정수 | `32` | 한 번에 추론할 이미지 수. 1 이상이어야 함 |
| `--workers` | 정수 | `0` | `DataLoader` 작업자 수. 0 이상이어야 함 |
| `--device` | `cpu`, `cuda`, `xpu` | `cpu` | 추론 장치 선택 |

실행 예시:

```powershell
python .\build_dinov2_index.py
```

```powershell
python .\build_dinov2_index.py --device cuda --batch-size 64 --workers 4
```

## 4. 전체 처리 흐름

1. 명령행 인자를 파싱한다.
2. 배치 크기와 작업자 수를 검증한다.
3. 이미지 폴더, CSV, 가중치 파일이 존재하는지 검사한다.
4. 이미지 폴더 아래의 지원 형식 파일을 재귀적으로 수집하고 경로순으로 정렬한다.
5. 실행 장치를 선택한다.
6. CSV를 읽어 품목일련번호별 메타데이터를 구성한다.
7. PyTorch Hub에서 DINOv2 ViT-S/14 모델 구조를 불러오고 로컬 가중치를 적용한다.
8. 이미지를 전처리하여 배치 단위로 임베딩을 추출한다.
9. 각 임베딩을 L2 정규화하고 CPU 메모리에 모은다.
10. 이미지 파일명과 CSV 메타데이터를 연결해 레코드를 만든다.
11. 임베딩, 레코드, 모델 정보를 `.pt` 파일로 저장한다.

## 5. 주요 구성 요소

### `PillImageDataset`

이미지 경로 목록을 받아 모델 입력 텐서를 생성하는 `Dataset`이다.

전처리 순서는 다음과 같다.

1. 이미지를 RGB로 변환한다.
2. Bicubic 보간으로 크기를 256에 맞춘다. `Resize(256)`이므로 짧은 변이 256이 되고 종횡비는 유지된다.
3. 중앙의 `224 × 224` 영역을 자른다.
4. 픽셀을 PyTorch 텐서로 변환한다.
5. ImageNet 평균과 표준편차로 정규화한다.

정규화 값:

```text
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

반환값은 `(이미지 텐서, 원본 목록 인덱스)`이지만, 현재 메인 추론 루프에서는 인덱스를 사용하지 않는다. 손상되었거나 PIL이 해석할 수 없는 이미지는 해당 경로를 포함한 `RuntimeError`를 발생시킨다.

### `select_device(requested)`

- `cpu`: CPU를 사용하며, `--device`를 생략했을 때 적용되는 기본값이다.
- `cuda`: CUDA GPU를 요구하며, 사용할 수 없으면 즉시 오류를 발생시킨다.
- `xpu`: Intel XPU를 요구하며, `torch.xpu`가 없거나 사용할 수 없으면 즉시 오류를 발생시킨다.

CUDA와 XPU는 자동 선택되지 않으며 각각 `--device cuda` 또는 `--device xpu`로 명시해야 한다.

### `load_checkpoint(path)`

가중치를 먼저 CPU로 읽는다.

- 최신 PyTorch에서는 `weights_only=True`를 사용한다.
- 해당 인자를 지원하지 않는 PyTorch 2.0 이전 버전에서는 일반 `torch.load()`로 재시도한다.
- 체크포인트 최상위에 `model` 키가 있으면 그 값을 실제 state dict로 사용한다.
- 최종 결과가 딕셔너리가 아니면 지원하지 않는 형식으로 판단한다.

### `load_model(weights, device)`

다음 모델 구조를 PyTorch Hub에서 가져온다.

```python
torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=False,
    trust_repo=True,
)
```

`pretrained=False`이므로 Hub에서는 모델 구조를 준비하고, 실제 가중치는 로컬 파일에서 읽는다. `strict=True`로 로드하므로 체크포인트의 키와 모델 구조가 정확히 일치해야 한다. 이후 평가 모드로 전환하고 선택한 장치로 이동한다.

PyTorch Hub 캐시에 DINOv2 소스가 없다면 최초 실행 시 `facebookresearch/dinov2` 저장소의 소스 코드를 내려받기 때문에 네트워크 연결이 필요할 수 있다.

### `load_metadata(csv_path)`

CSV를 `utf-8-sig`로 읽기 때문에 UTF-8 BOM이 있는 파일도 처리할 수 있다. 필수 컬럼은 다음 세 개다.

- `품목일련번호`
- `품목명`
- `의약품제형`

빈 품목일련번호 행은 무시한다. 같은 품목일련번호가 여러 행에 존재하면 품목명과 제형을 각각 집합으로 모은다.

- 고유 값이 하나이면 그 값을 사용한다.
- 품목명이 여러 종류이면 `(CSV 품목명 중복)`을 사용한다.
- 제형이 여러 종류이면 `(CSV 제형 충돌)`을 사용한다.

빈 문자열도 집합에 포함되므로, 한 품목 ID에 빈 값과 정상 값이 함께 있으면 중복 또는 충돌로 판정될 수 있다.

## 6. 임베딩 생성

`DataLoader`는 입력 경로의 정렬 순서를 유지하기 위해 `shuffle=False`로 설정된다. CUDA 사용 시 `pin_memory=True`가 되며, 장치 전송에는 `non_blocking=True`를 사용한다.

추론은 그래디언트를 만들지 않는 `torch.inference_mode()` 안에서 실행된다.

```python
embeddings = F.normalize(model(images).float(), p=2, dim=1)
```

모델 출력은 `float32`로 변환한 후 각 행을 L2 norm 1이 되도록 정규화한다. 따라서 이후 두 임베딩의 내적은 코사인 유사도와 같으며, 이미지 검색에서 바로 사용할 수 있다.

각 배치의 결과는 CPU로 이동해 리스트에 보관하고 마지막에 `torch.cat(..., dim=0)`으로 결합한다. 전체 임베딩 텐서의 형태는 다음과 같다.

```text
[이미지 수, 384]
```

## 7. 이미지와 CSV의 연결 규칙

품목일련번호는 이미지 파일의 확장자를 제외한 이름에서 첫 번째 밑줄 앞부분을 가져와 결정한다.

```text
파일명: 200808876_front.jpg
item_id: 200808876
```

밑줄이 없다면 파일명 전체가 품목일련번호가 된다. 파일명에서 얻은 ID가 CSV에 없으면 품목명과 제형에 `(CSV 매칭 없음)`을 저장한다.

`side` 값은 이미지의 바로 위 부모 폴더 이름을 소문자로 바꾼 값이다. 예를 들어 `pill_images_original/front/example.jpg`의 `side`는 `front`가 된다. 따라서 디렉터리 구조가 앞면·뒷면 의미와 일치해야 한다.

## 8. 출력 파일 구조

출력은 `torch.save()`로 저장한 딕셔너리다.

```python
{
    "format_version": 1,
    "model_name": "dinov2_vits14",
    "embedding_dim": 384,
    "embeddings": Tensor[N, 384],
    "records": list[dict],
}
```

각 `records` 원소는 같은 위치의 임베딩을 설명한다.

```python
{
    "path": "이미지의 절대 경로",
    "item_id": "파일명에서 추출한 품목일련번호",
    "product_name": "CSV 품목명 또는 상태 메시지",
    "shape": "CSV 의약품제형 또는 상태 메시지",
    "side": "부모 폴더 이름(소문자)",
}
```

즉, `embeddings[i]`와 `records[i]`는 동일한 이미지를 가리킨다. 이미지 경로를 절대 경로로 저장하므로 프로젝트나 이미지 폴더를 다른 위치로 옮기면 저장된 경로가 더 이상 유효하지 않을 수 있다.

출력 파일 읽기 예시:

```python
import torch

index = torch.load("dinov2_pill_index.pt", map_location="cpu", weights_only=False)
embeddings = index["embeddings"]
records = index["records"]

print(embeddings.shape)
print(records[0])
```

## 9. 검증 및 예외 처리

스크립트는 다음 상황에서 명시적으로 중단한다.

- `batch-size < 1` 또는 `workers < 0`
- 이미지 폴더가 없음
- CSV 파일이 없음
- 가중치 파일이 없음
- 지원 확장자의 이미지가 하나도 없음
- `--device cuda`를 지정했지만 CUDA를 사용할 수 없음
- `--device xpu`를 지정했지만 Intel XPU를 사용할 수 없음
- CSV 필수 컬럼이 없음
- 체크포인트가 지원하지 않는 형식임
- 모델과 체크포인트의 키가 정확히 일치하지 않음
- 이미지를 읽거나 해석할 수 없음

출력 부모 폴더는 없으면 자동으로 생성된다.

## 10. 특성과 주의점

- 이미지 중앙을 자르므로 알약이 가장자리나 중앙 밖에 있으면 중요한 부분이 손실될 수 있다.
- 손상 이미지 하나가 발견되면 전체 작업이 중단되며, 해당 이미지를 건너뛰는 로직은 없다.
- 모든 배치 임베딩을 메모리에 보관한 뒤 한 번에 합치므로 이미지 수가 매우 많으면 CPU 메모리 사용량이 커진다.
- 이미지 정렬 순서가 임베딩과 레코드의 대응 관계를 결정한다.
- CSV 중복 처리에서 동일 ID에 값이 둘 이상이면 임의의 값을 선택하지 않고 상태 문자열을 기록한다.
- `trust_repo=True`로 외부 Hub 저장소 코드를 신뢰해 로드하므로 실행 환경의 보안 정책을 확인할 필요가 있다.
- 출력에 모델 가중치의 버전이나 해시가 기록되지 않으므로, 같은 모델명이라도 다른 체크포인트로 생성한 인덱스를 구별하기 어렵다.
- `embedding_dim`은 모델 출력에서 계산하지 않고 `384`로 고정되어 있으므로 모델 구조를 변경할 경우 함께 수정해야 한다.

## 11. 요약

이 스크립트의 핵심 목적은 알약 이미지 전체를 DINOv2 특징 공간으로 변환해 코사인 유사도 검색에 사용할 수 있는 인덱스를 만드는 것이다. 파일명 규칙으로 품목 ID를 얻고 CSV 메타데이터를 결합하며, 최종 결과에서 임베딩과 레코드는 같은 인덱스로 일대일 대응한다.
