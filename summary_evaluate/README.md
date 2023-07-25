# summary_evaluate: 리뷰 요약 모델 평가를 위한 폴더

## 구성 파일

- **keyword_extractor.py** > `class KeywordExtractor`
  - 요약문에서 키워드를 추출하기 위한 클래스
- **summary_inference.py** > `inference()`
  - (1) 생성한 요약문과 (2) 소요 시간 반환
- **summary_evaluate.py** > `evaluate()`
  - (1) 전체 테스트 데이터셋에 대한 평가 점수 반환
  - (2) 각 데이터에 대한 평가 점수 반환
- **summary_scores.py**
  - 여러 평가 지표 함수 모음
    |함수|평가 지표|
    |---|---|
    |`get_keyword_score()`|(1) 키워드 기반 F2 점수<br>(2) 키워드 기반 Summary-level ROUGE-L|
    |`get_length_penalty()`|(1) 길이 페널티<br>(2) 길이 차이|
    |`get_sts_score()`|유사도|
- **summary_utils.py**
  - `save_evaluation()`: 평가 결과를 json 파일로 저장하는 함수
- **templates/**
  - 요약 모델 추론에 사용하는 프롬프트 형식을 저장한 json 파일을 모아둔 폴더

## 요약 모델 평가 방법

### 준비물

1. 테스트 데이터셋: 테스트 데이터셋을 불러와서 형식에 맞게 변환하는 코드
    - 테스트 데이터셋 전처리 및 필터링 잊지 말 것
2. 요약 모델 결과물:
    - (1) 평가할 요약 모델이 테스트 데이터셋에 대해 생성한 요약문: `List[str]`
    - (2) 각 요약문을 생성하는데 걸린 시간: `List[int]`

### 테스트 데이터셋 형식

  - `inference()`, `evaluate()`에 넘겨주는 테스트 데이터셋은 다음 dict를 원소로 갖는 리스트
    ```
    {
        "id": int - 고유한 데이터 인덱스,
        "prod_name": 상품명,
        "review": str - 요약할 리뷰 데이터,
        "summary": str - 레퍼런스 요약문,
    }
    ```

### summary_evaluate.py 사용법

1. `__main__`의 `test_dataset`에 테스트 데이터셋을 저장
2. `__main__`의 `pred, test`에 요약 모델 생성물과 소요시간 저장
3. 평가 예시는 주석 처리
4. **summary_evaluate.py** 실행

### 평가 결과물

  ```
  {
      "total": { // 전체 데이터셋에 대한 평가 결과
          "f2_penalty": 평균 (f2 점수 + 길이 페널티), 
          "f2": 평균 f2 점수, 
          "time": 평균 소요 시간, 
          "f2": 평균 키워드 기반 F2 점수,
          "rougeLsum": 평균 키워드 기반 Summary-level Rouge-L 점수,
          "sts_score": 평균 유사도 점수
          }, 
      "results": [ // 각 데이터에 대한 평가 결과
          {
              "id": 데이터 번호, 
              "review": 요약한 리뷰,
              "ref": 레퍼런스 요약문,
              "pred": 요약 모델 생성 결과,
              "time": 생성 소요 시간, 
              "ref_keywords": 레퍼런스 요약문에서 추출한 키워드,
              "pred_keywords": 요약 모델 생성 결과에서 추출한 키워드,
              "f2": 키워드 기반 F2 점수,
              "rougeLsum": 키워드 기반 Summary-level Rouge-L 점수,
              "sts_score": 유사도 점수, 
              "length_diff": 생성 결과 길이 - 답 길이, 
          }, ...]
  }
  ```

- `f2, rougeLsum, sts_score`에는 100을 곱해서 저장
- `time`은 소요 시간 리스트가 주어지지 않았다면 `-1.0`으로 초기화
