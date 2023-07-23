# generate

## 구성 파일

- **create_input_data.py**
  - 요약문을 제외한 데이터셋 제작
- **openai_generate.py**
  - OpenAI API 사용해서 요약문 생성
- **merge_output_data.py**
  - 생성한 요약문을 데이터셋에 넣어서 완성

## 사용법

- 각 코드에서 입력, 출력 파일 경로 수정하기
- OpenAI API key가 들어있는 .env 파일 위치에서 실행
- 실행 순서
    ```bash
    python create_input_data.py
    python openai_generate.py
    python merge_output_data.py
    ```

