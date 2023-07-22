import sys
sys.path.append('/opt/ml/input/level3_nlp_finalproject-nlp-07')
from utils.preprocess import clean_text, get_no_space_length

import re
import pandas as pd
from tqdm import tqdm
import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

class ReviewSummaryData:
    def __init__(self, data_id, prod_name):
        self.id = data_id
        self.prod_name = prod_name
        self.review = ""
        self.summary = ""
        self.review_no_space_length = 0
        self.review_tokens = 0
    
    def add_review(self, review):
        self.review += ". " + clean_text(review)
        self.review_no_space_length = get_no_space_length(self.review)
        self.review_tokens = num_tokens_from_string(self.review)
    
    def get_dict(self):
        
        # 맨 앞 '. ' 제거, 맨 뒤 '.' 붙이기
        self.review = self.review[2:] + "."
        self.review_no_space_length = get_no_space_length(self.review)
        self.review_tokens = num_tokens_from_string(self.review)
        
        return {
            "id": self.id,
            "prod_name": self.prod_name,
            "review": self.review, 
            "summary": self.summary,
            "review_no_space_length": self.review_no_space_length,
            "review_tokens": self.review_tokens
        }

def initialize_summary_dataset(df: pd.DataFrame, min_tokens = 500, max_tokens = 2000):
    """리뷰 요약 데이터셋 형태를 만든다. 입력 데이터셋과 길이가 짧아서 버려진 데이터셋을 반환한다."""
    
    REVIEW_ID = 'review_id'
    PROD_NAME = 'prod_name'
    CONTEXT = 'context'
    
    column_names = [REVIEW_ID, PROD_NAME, CONTEXT]
    for cn in column_names:
        assert cn in df.keys(), f"DataFrame should have following column names: {column_names}"
    
    data_list = []
    del_list = []
    
    cur = None
    
    for idx, item in tqdm(df.iterrows(), total=len(df), desc="Concating reviews"):
        if not isinstance(item[CONTEXT], str) or len(item[CONTEXT]) == 0: 
            continue        
        
        if not cur: # 처음에만 해당
            new_data_id = len(data_list) + 1
            cur = ReviewSummaryData(new_data_id, item[PROD_NAME])
        elif item[PROD_NAME] != cur.prod_name: # 다른 상품
            if num_tokens_from_string(cur.review) >= min_tokens: 
                data_list.append(cur.get_dict())
            else:
                del_list.append(cur.get_dict())
                
            new_data_id = len(data_list) + 1
            cur = ReviewSummaryData(new_data_id, item[PROD_NAME])
        
        sents = [s for s in item[CONTEXT].split(".") if len(s) > 0]
            
        for se in sents:
            if cur.review_tokens + num_tokens_from_string(se) > max_tokens:
                if cur.review_tokens >= min_tokens:
                    data_list.append(cur.get_dict())
                else:
                    del_list.append(cur.get_dict())
            
                new_data_id = len(data_list) + 1
                cur = ReviewSummaryData(new_data_id, item[PROD_NAME])        
            
            cur.add_review(se)    

    # 마지막에 저장 안 된 것
    if cur.review_tokens >= min_tokens:
        data_list.append(cur.get_dict())
    else:
        del_list.append(cur.get_dict())
        
    return data_list, del_list

if __name__ == "__main__":
    
    from filtering.review_filter import ReviewFilter
    from summary.summary_utils import clean_text
    filter = ReviewFilter()

    # 전처리 및 필터링 된 리뷰 DB 테이블 데이터.  각 리뷰 문장은 마침표로 끝나야 함.
    INPUT_CSV_PATH = "/opt/ml/input/data/reviews_ver31_clean.csv"
    # 합쳐진 리뷰를 저장할 경로
    OUTPUT_JSON_PATH = "/opt/ml/input/data/summary_v3.5_1500.json"
    # 길이 조건에 맞지 않아 삭제된 리뷰를 저장할 경로
    DELETED_OUTPUT_JSON_PATH = "/opt/ml/input/data/summary_v3.5_1500_deleted.json"
    
    print("Input:", INPUT_CSV_PATH)
    print("Output:", OUTPUT_JSON_PATH)
    print("Deleted:", DELETED_OUTPUT_JSON_PATH)

    df = pd.read_csv(INPUT_CSV_PATH)
    df = df[["review_id", "prod_name", "context"]]
    
    data_list, del_list = initialize_summary_dataset(df, 500, 1500)
    
    print(f"Number of data: {len(data_list)}")
    print(f"Number of deleted data: {len(del_list)}")

    import json
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False)
        
    with open(DELETED_OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(del_list, f, ensure_ascii=False)
        
    print(f"Files saved at: {OUTPUT_JSON_PATH}, {DELETED_OUTPUT_JSON_PATH}")