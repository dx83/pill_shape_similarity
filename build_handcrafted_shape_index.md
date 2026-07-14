# build_handcrafted_shape_index.py 설명

`build_handcrafted_shape_index.py`는 `pill_images_original`의 알약 이미지와 CSV의 품목 정보를 결합하여 제형 추론용 인덱스를 만든다. 딥러닝 모델을 학습하지 않고 `shape_features.py`의 수작업 형상 임베딩을 사용한다.

## 기본 실행

```powershell
python .\build_handcrafted_shape_index.py
```

기본 입력과 출력은 다음과 같다.

| 구분 | 경로 |
|---|---|
| 이미지 | `pill_images_original` |
| CSV | `csv\OpenData_pill_final_cleaned_updated.csv` |
| 인덱스 | `handcrafted_shape_index.npz` |
| 보고서 | `handcrafted_shape_build_report.json` |

경로는 `--images`, `--csv`, `--output`, `--report` 옵션으로 바꿀 수 있다. 빠른 동작 확인에는 `--limit 500`처럼 처리 이미지 수를 제한할 수 있다.

## 집계 과정

```text
이미지별 56차원 특징
  → 같은 품목 이미지의 특징 중앙값
  → robust scaling 및 특징 그룹 가중치
  → 같은 제형 품목 벡터의 중앙값
```

평균 대신 중앙값을 사용하여 앞·뒷면 중 한 이미지의 윤곽이 잘못 추출되더라도 영향을 줄인다. 인덱스에는 제형별 중심 프로토타입뿐 아니라 품목별 대표 벡터, 품목일련번호, 품목명, 제형, 이미지 수도 저장한다. 추론기는 품목별 대표 벡터를 exemplar prototype으로 이용하여 클래스 내부의 다양한 형상을 보존한다.

## 제외 규칙

- 같은 품목일련번호에 서로 다른 `의약품제형`이 있으면 제형 충돌로 보고 해당 품목 전체를 제외한다.
- 의약품제형이 없거나 CSV에 매칭되지 않는 이미지는 제외한다.
- 알약 마스크나 유효한 특징을 만들 수 없는 이미지는 실패 목록에 기록하고 나머지 데이터 처리는 계속한다.

`handcrafted_shape_build_report.json`에는 전체 이미지 수, 성공·실패 수, 제외된 충돌 품목, 클래스별 품목·이미지 수, 실패 이미지 경로와 원인이 기록된다.

## 현재 전체 빌드 결과

- 인덱싱 제형: 29개
- 인덱싱 품목: 27,345개
- 특징 추출 성공: 54,282장
- 특징 추출 실패: 1장
- 제형 충돌 제외: 2품목, 18장

원본 이미지나 CSV는 수정하지 않는다.
