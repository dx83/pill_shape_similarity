# predict_handcrafted_shape.py 설명

`predict_handcrafted_shape.py`는 새 알약 이미지에서 형상 임베딩을 계산하고 `handcrafted_shape_index.npz`의 프로토타입과 거리를 비교하여 의약품제형을 추측한다.

## 기본 실행

```powershell
python .\predict_handcrafted_shape.py .\test_images\200907150583201.jpg
```

상위 결과 수를 바꾸려면 `--top-k`를 사용한다.

```powershell
python .\predict_handcrafted_shape.py IMAGE_PATH --top-k 5
```

프로그램 연동용 JSON 결과는 `--json`으로 출력한다.

```powershell
python .\predict_handcrafted_shape.py IMAGE_PATH --top-k 5 --json
```

## 입력 형태

- 투명 배경의 알약 한 장
- 배경이 있는 JPG 한 장
- 앞면과 뒷면이 같이 있는 원본 제품 이미지

한 이미지에서 여러 알약이 검출되면 각 알약의 결과를 계산하고, 같은 품목 프로토타입까지의 거리를 알약 후보 전체에서 중앙값으로 집계한다. `--max-objects`로 최대 검출 개수를 바꿀 수 있다.

## 거리 계산

주 순위는 각 제형에 속한 **품목별 exemplar prototype 중 가장 가까운 거리**로 결정한다. 제형마다 하나의 평균 벡터만 사용하면 사과형처럼 클래스 내부 변형이 많은 제형이 평균 과정에서 뭉개질 수 있기 때문이다.

각 결과에는 다음 정보가 포함된다.

- `distance` — 가장 가까운 같은 제형 품목 프로토타입까지의 거리. 작을수록 유사하다.
- `class_centroid_distance` — 제형 전체 중심 프로토타입까지의 보조 거리.
- `relative_score` — 결과 간 상대 순위를 보기 위한 값이며 확률이 아니다.
- `within_training_range` — 학습 품목이 분포한 제형 반경 안인지 나타내는 참고 값.
- `nearest_item_id`, `nearest_item_name` — 가장 유사한 실제 학습 품목.
- `top_margin` — 2위 거리에서 1위 거리를 뺀 값. 작으면 두 제형의 구분이 애매하다.

## 결과 해석 주의사항

이 프로그램의 점수는 통계적으로 보정된 확률이 아니다. 특히 장방형과 타원형, 땅콩형과 8자형처럼 경계가 연속적인 제형은 1위와 2위 거리가 비슷할 수 있다. 이런 경우 Top-1을 확정값으로 사용하지 말고 Top-k 후보 또는 사용자 확인 절차를 사용하는 것이 안전하다.

현재 테스트 이미지 3개에서는 CSV 제형인 장방형, 사과형, 땅콩형이 각각 1위로 확인됐다.
