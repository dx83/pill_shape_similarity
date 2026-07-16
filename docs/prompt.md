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


다음 이미지에 알약만 크롭해서 _crop 파일명 붙여서 동일한 폴더에 넣어줘
\test_images\1번.jpg
\test_images\3번.jpg
\test_images\4번.jpg


# ppt 발표 자료 작성

## 참고 파일
build_handcrafted_shape_index.md
predict_handcrafted_shape.md
shape_features.md
평가.md (이미 자료를 정리한 파일)

## 요구 사항
1. 평가.md에 이미 정리를 했지만 필요하면 나머지 파일도 참고 가능
2. ppt 3장 분량의 내용을 ppt.md 로 작성
    - ppt 3장에 들어갈 내용과 레이이웃(디자인) 설명
    - 실제 발표할 때 스크립트
    - 중간에 들어갈 내용이므로 ppt 표지나 마지막 페이지 같은 거 고려할 필요없음
3. 배정된 양이 3장 분량이므로 한계와 실패의 내용은 쓰지 않는다.
4. poc 라는 단어는 쓰지 않는다.



# ppt 파일 작성
1. ppt 폴더에 필요한 모든 자료가 있다.
2. ppt.note.md 파일을 바탕으로 proj2_ssm.pptx 작성
3. ppt 템플릿은 ppt_template_proj2.pptx를 사용
4. ppt는 3장만 만들면 된다.


# 이미지 교체
1. ppt 폴더에 있는 proj2_ssm.pptx 의 첫 번째 페이지에 들어가는 이미지 shape_extraction_process.png 를 수정하고 싶다.
2. 기존 이미지에는 사과형이 들어가 있는데 'C:\Workspace\deeplearning\pill_shape_similarity\test_images\A11AOOOOO100201.jpg' 이 이미지로 교체하고 싶다.
3. 기존 이미지의 내용 형식은 지키면서 사과형 알약만 물방울3 알약으로 교체시켜줘
4. 수정한 이미지는 shape_extraction_process_.png 로 만들어줘



# 요구사항
니가 크롭한 이미지들인데 크롭만 해야하는데 가공이 너무 들어가서 알약 같지가 않다.
"C:\Workspace\deeplearning\pill_shape_similarity\ppt\shape_apple.png"
"C:\Workspace\deeplearning\pill_shape_similarity\ppt\shape_peanut.png"
"C:\Workspace\deeplearning\pill_shape_similarity\ppt\shape_rectangular.png"

다음 원본 이미지들에서 위와 같은 형식으로 다시 크롭해줘

"C:\Workspace\deeplearning\pill_shape_similarity\test_images\200909230007701.jpg"
이미지내에서 왼쪽 알약만 크롭해서 다른 가공 없이 ppt\shape_apple_.png 로 생성

"C:\Workspace\deeplearning\pill_shape_similarity\test_images\200907150583201.jpg"
이미지내에서 왼쪽 알약만 크롭해서 다른 가공 없이 ppt\shape_rectangular_.png

"C:\Workspace\deeplearning\pill_shape_similarity\test_images\201110240001401.jpg"
이미지내에서 왼쪽 알약만 크롭해서 다른 가공 없이 ppt\shape_peanut_.png


# 추가 수정
니가 만든 이미지들 배경 수정이 필요하다
이미지 사이즈를 800x500 으로 동일하게 맞추고 알약 이미지 자체는 크기를 늘리거나 하지마
나머지를 배경으로 채우는데 "C:\Workspace\deeplearning\pill_shape_similarity\ppt\shape_apple.png"와 동일한 형식으로 채워줘
수정할 이미지들
    - ppt/shape_apple_.png
    - ppt/shape_rectangular_.png
    - ppt/shape_peanut_.png
수정한 후 파일명 뒤에 bg 를 붙여서 저장해줘
