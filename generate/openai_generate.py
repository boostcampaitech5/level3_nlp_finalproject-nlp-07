import sys
sys.path.append('/opt/ml/input/level3_nlp_finalproject-nlp-07')
from utils.preprocess import clean_text, get_no_space_length
from utils.prompter import Prompter

from typing import List
import openai
import os
from dotenv import load_dotenv
import json
import time
from tqdm import tqdm
import copy

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 프롬프트 형식 파일 절대 경로
prompter = Prompter("/opt/ml/input/level3_nlp_finalproject-nlp-07/generate/templates/openai_v3.5.json")

def generate_text(prod_name, review):
    system_prompt = "너는 상품 리뷰에서 상품에 대한 특성을 추출해 요약해주는 모델이야."
    review = clean_text(review)
    user_prompt = prompter.generate_prompt(review=review)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        # max_tokens=500,
    )

    result = response.choices[0].message.content
    token_cnt = response.usage
    stop_reason = response.choices[0].finish_reason

    item = {
        "text": result,
        "token_cnt": token_cnt,
        "stop_reason": stop_reason,
        "no_space_length": get_no_space_length(result)
    }

    return item

def main(input_data_path: str, output_data_paths: List[str]):
    
    # 리뷰 데이터 들어있는 파일 경로
    with open(
        input_data_path,
        "r", encoding="utf-8"
    ) as f:
        data_list = json.load(f)

    # 중간 결과물 저장할 파일 경로 인덱스
    save_idx = 0

    result_list = []
    prompt_tokens = 0
    completion_tokens = 0

    # 기존에 만들다가 중단된 파일이 있는지 확인
    for idx, path in enumerate(output_data_paths):
        dir_path = os.path.split(path)[0]
        os.makedirs(dir_path, exist_ok=True)        
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)
            if len(r) > len(result_list): 
                result_list = copy.deepcopy(r)
                save_idx = idx
    
    if len(result_list) > 0:
        print("Recovering from previous file:", output_data_paths[save_idx])
        print("Data already generated:", len(result_list))
    else:
        print("Reading from file:", input_data_path)
        
    if len(result_list) == len(data_list):
        print("All data is generated.\nChange data_path or delete existing files")
        return 0

    for item in result_list:
        prompt_tokens += item["token_cnt"]["prompt_tokens"]
        completion_tokens += item["token_cnt"]["completion_tokens"]
        
    data_list = data_list[len(result_list):]
    save_idx = (save_idx + 1) % len(output_data_paths)

    print("Data to generate:", len(data_list))

    start = time.time()

    save_cnt = 1  # 저장 주기. save_cnt 개의 데이터 생성한 뒤 저장

    for item in tqdm(data_list, desc="Generating text", total=len(data_list)):
        
        # For testing
        # if len(result_list) >= 5:
        #     break
        
        result = generate_text(item["prod_name"], item["review"])
        result_list.append(result)

        prompt_tokens += result["token_cnt"]["prompt_tokens"]
        completion_tokens += result["token_cnt"]["completion_tokens"]

        save_cnt -= 1
        if save_cnt == 0:
            with open(output_data_paths[save_idx], "w", encoding="utf-8") as f:
                json.dump(result_list, f, ensure_ascii=False)

            save_idx = (save_idx + 1) % len(output_data_paths)
            save_cnt = 10
            
       

    # save_cnt 횟수 남아서 저장 안 된 것도 저장
    with open(output_data_paths[save_idx], "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False)

    end = time.time()
    print(f"time: {end - start:.5f} sec")
    print("Final data saved at:", output_data_paths[save_idx])
    print("prompt_tokens:", prompt_tokens)
    print("completion_tokens:", completion_tokens)
    print("total_tokens:", prompt_tokens + completion_tokens)
    
    return 0

if __name__ == "__main__":
    
    # # 리뷰 데이터 들어있는 파일 경로
    INPUT_DATA_PATH = "/opt/ml/input/data/v3.5/summary_v3.5_1500.json"
    # 중간 결과물 저장할 파일 경로
    OUTPUT_DATA_PATHS = [
        "/opt/ml/input/data/v3.5/prompt_openai_1500_0.json",
        "/opt/ml/input/data/v3.5/prompt_openai_1500_1.json",
    ]
    
    error_cnt = 0
    wait_sec = 1 # 에러가 연속되면 기다리는 시간 증가
    while True:
        try:
            ret_code = main(INPUT_DATA_PATH, OUTPUT_DATA_PATHS)
            if ret_code == 0: break
            wait_sec = 1
        except (
            openai.error.AuthenticationError,
            openai.error.InvalidRequestError
            )as e:
            print(f"OpenAI API Error: {e}")
            print("Terminate.")
            break
        except(
            openai.error.ServiceUnavailableError,
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError
        ) as e: # wait and continue
            error_cnt += 1
            print(f"OpenAI API Error: {e}")
            print(f"{error_cnt}th error! Trying again.")
            time.sleep(wait_sec)
            wait_sec += 1
        except Exception as e:
            print("Error:", e)
            print("Terminate.")
            break
        
