# 개요
- csv를 참고하여 이미지 데이터의 개수를 보여주는 python 스크립트

# 세부 사항
- csv 폴더에 OpenData_pill_final_cleaned_updated.csv 가 있다.
- pill_images_original 폴더에 실제 데이터가 있다.
- 의약품제형 컬럼의 각 제형별로 실제 몇개가 있는지 알고 싶다.
- shape_data_view.py 파일로 만들것.
- 원본 데이터의 수정이나 가공이 있으면 안된다.



# 작업 개요
니가 추천한 방법 사용 : EFD를 포함한 수작업 형상 임베딩 + 클래스 프로토타입 거리 비교

# 이미지 데이터
\pill_images_original 폴더

# 해당 csv 파일의 품목일련번호, 품목명, 의약품제형 컬럼
\csv\OpenData_pill_final_cleaned_updated.csv

# 참고 사항
pill_images_original 에는 알약 이미지가 있고, csv 파일에는 해당 이미지의 제형이 나와 있다.
shape_data_view.py 실행시키면 제형마다 개수가 나온다. 여기서 제형 충돌 항목은 임베딩에서 제외한다.

# 요구 사항
1 알약 이미지의 제형 임베딩하는 py 파일 생성 
2 알약 이미지의 제형을 추측하는 py 파일 생성
3 필요에 따라 추가 파일 생성 가능
4 모든 파일이 완성되면 각 파일마다 각 파일을 설명하는 .md 파일 생성
