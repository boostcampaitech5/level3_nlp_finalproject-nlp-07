# summary: 리뷰 요약 모델에 대한 폴더

## 구성 파일

- **summary_inference.py** > `inference()`
  - (1) 생성한 요약문과 (2) 소요 시간 반환
- **summary_evaluate.py** > `evaluate()`
  - (1) 전체 테스트 데이터셋에 대한 평가 점수 반환
  - (2) 각 데이터에 대한 평가 점수와 로그 반환
- **summary_scores.py**
  - 여러 평가 지표 함수 모음
    |함수|평가 지표|
    |---|---|
    |`get_f2_score()`|f2 점수|
    |`get_length_penalty()`|길이 페널티|
    |`get_sts_score()`|유사도|
- **summary_utils.py**
  - 전처리 함수들
  - `save_evaluation()`: 평가 결과를 json 파일로 저장하는 함수
  - `class Prompter`: 요약 모델 입력 프롬프트 제작을 위한 클래스
- **templates/**
  - 프롬프트 형식을 저장한 json 파일을 모아두는 폴더

## 요약 모델 평가 방법

- 테스트 데이터셋 형식
  - `inference()`, `evaluate()`에 넘겨주는 테스트 데이터셋은 다음 dict를 원소로 갖는 리스트
    ```
    {
        "id": int - 고유한 데이터 인덱스,
        "review": str - 요약할 리뷰 데이터,
        "summary": str - 답 요약문,
        "keywords": List[str] - f2 점수에 사용될 키워드 리스트
    }
    ```
- 요약 모델을 평가하는 방법
  - (1) **templates/** 폴더에 프롬프트 형식 json 파일을 추가
  - (2) 테스트 데이터셋을 형식에 맞춰 준비하고, `__main__` 속 `test_dataset` 변수에 저장
  - (3) 평가할 요약 모델의 이름을 `MODEL` 변수에 저장
  - (4) `inference(..., promt_template_name)`에서 `prompt_template_name`에 (1)에 추가한 파일 이름 (확장자 제외) 명시
  - (5) 요약 모델 프롬프트를 제작할 때, `review` 외 다른 키를 사용한다면, `inference()` 코드 수정 필요
  - (6) **templates/** 폴더와 같은 디렉토리에서 `summary_evaluate.py` 파일을 실행
- 평가 결과물
```
{
    "total": { // 전체 데이터셋에 대한 평가 결과
        "f2_penalty": 평균 (f2 점수 + 길이 페널티), 
        "f2": 평균 f2 점수, 
        "time": 평균 소요 시간, 
        "sts_score": 평균 유사도 점수
        }, 
    "results": [ // 각 데이터에 대한 평가 결과
        {
            "id": 데이터 번호, 
            "f2_penalty": f2 점수 + 길이 페널티, 
            "f2": f2 점수, 
            "time": 생성 소요 시간, 
            "sts_score": 답과의 유사도 점수, 
            "length_diff": 생성 결과 길이 - 답 길이, 
            "log": keywords 탐색 결과, 존재하는 keywords 개수, keywords가 하나도 없는 문장 개수 및 리스트에 대한 로그, 
            "input": 요약할 리뷰 데이터, 
            "output": 생성한 요약문
        }, ...]
}
```

