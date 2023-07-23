# filtering: 리뷰 필터링 관련 코드

## 구성 파일

- **cluster.ipynb**
  - 클러스터 기반 필터링
  - 임의의 클러스터 중심 설정 eg) 평가, 조리
  - csv 파일에 있는 모든 리뷰 문장을 분리해서 클러스터링 후, 원하는 클러스터에 포함되어 있지 않은 문장 필터링

- **notebook_filtering.ipynb**
  - 룰 기반 필터링 실험용 코드
  - 상품 별 필터링 전후 리뷰 길이 확인

- **review_filter.py** > `class ReviewFilter`
  - 룰 기반 필터링을 위한 클래스
  - `filter_text()`: 하나의 문자열 속 문장들을 필터링한 후 반환
  - `get_filter_score()`: 전후 길이를 받아 필터링 비율을 반환

## 룰 기반으로 필터링 하는 법

### 필요 라이브러리 

```
pip install kiwipiepy
```

### 사용법

필터링한 텍스트를 미리 전처리하되, 마침표(`.`)는 문장 분리에 도움을 주므로 전처리 하지 않는 것을 추천합니다. 필터링 결과는 각 문장 뒤에 마침표가 붙여서 나옵니다.

``` python
from review_filter import ReviewFilter
from utils.preprocess import clean_text, get_no_space_length

text = "필터링 할 문자열"
text = clean_text(text, remove_dot=False)

filter = ReviewFilter()

filtered_text = filter.filter_text(text, verbose=True, join=True)
print(filtered_text)
print(filter.get_filter_score(get_no_space_length(text), get_no_space_length(filtered_text), verbose=True))
```

### 필터링 룰

1. kiwipiepy의 `Kiwi.split_into_sents()` 로 각 리뷰를 문장으로 분리
2. 중복된 문장이 있으면 처음 것 제외 필터링
3. 문장 길이 MIN_LEN ~ MAX_LEN 자 외 필터링 (디폴트 10, 200)
4. 정규표현식을 이용해서 특정 단어가 들어간 문장 필터링
```python
del_patterns = [
    "조리", "해동", "요리", '추가','[센중약]불', "넣", "헹", "방법", # 레시피
    '작성', '도움', '내돈내산','리뷰','안녕','보답','감사','눌러','좋은 하루', "후기", # 리뷰 끝맺음
    "(유통)?기한", "보관",
    "(재)?구매", "(재)?구입",
    "배달", "배송", "로켓", "프레시",'주문',
]
```
