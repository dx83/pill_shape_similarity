

https://github.com/facebookresearch/dinov2

## 필요한 라이브러리

torch
torchvision
pillow
numpy
pandas

설치 명령은 GPU/CUDA 환경에 따라 PyTorch 버전이 달라지므로 먼저 GPU 환경을 확인해야 합니다.

## 실제로 만들어야 할 코드

최소한 다음 두 스크립트면 작동합니다.

- build_dinov2_index.py: 54,301개 이미지 임베딩을 추출하여 저장
- search_dinov2_topk.py: 검색 이미지와 코사인 유사도가 높은 이미지 출력

CSV는 모델 실행 자체에는 필요하지 않습니다. 검색 결과에 품목명, 의약품제형, 색상 등을 표시할 때 품목일련번호로 연결하는 용도입니다.

그리고 처음부터 학습할 필요는 없습니다. 우선 받은 ViT-S/14 가중치로 Top-K 결과를 확인하고, 검색 품질이 부족할 때 알약 데이터로 파인튜닝하면 됩니다.
