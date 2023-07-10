import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse
import re
import json
import os
import evaluate
from tqdm import tqdm
from datetime import datetime
from pytz import timezone


# 특수문자, 초성어 제거
def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s0-9a-zA-Z]", " ", sent)
    sent_clean = re.sub("[ㄱ-ㅎㅏ-ㅣ]+", "", sent_clean)
    sent_clean = " ".join(sent_clean.split())
    sent_clean = sent_clean.strip()
    return sent_clean

# 입력 프롬프트 제작
def build_text(keyword, review):
    if args.format == "my":
        text = (  # multi-line string
            f"### 명령어: 다음의 {keyword}에 대한 리뷰를 특성 세부설명 형식으로 요약하세요\n\n"
            "### 리뷰: " + review.strip() + "\n\n"
            "### 요약:"
        )
    elif args.format == "koalpaca":
        text = (
            f"### 질문: 다음의 {keyword}에 대한 리뷰를 특성 세부설명 형식으로 요약하세요.\n\n"
            "" + review.strip() + "\n\n"
            "### 답변:"
        )

    return text


def inference(model, tokenizer, test_data: pd.DataFrame):
    # MODEL = "models/0_e5" # 가장 결과 좋음
    # MODEL = "models/koalpaca-ft_0-myFormat"
    # MODEL = "models/0_noTag"
    # MODEL = "models/e10"
    # MODEL = "models/e5lr1/"

    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    def ask(input_text, is_input_full=False):
        ans = pipe(
            input_text,
            do_sample=True,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
            eos_token_id=2,
        )
        result = ans[0]["generated_text"]
        return result

    rouge = evaluate.load("rouge")

    result_list = []
    preds = []
    for item in tqdm(test_data.itertuples(), desc="Making predictions", total=len(test_data)):
        clean_review = clean_text(item.input)
        input_text = build_text(item.search_name, item.input)
        generated_text = ask(input_text)
        generated_text = clean_text(generated_text)
        result_item = {
            "id": item.id,
            "input": clean_review,
            "output": generated_text,
            "score": None,
        }
        preds.append(generated_text)
        result_list.append(result_item)

    total_score = rouge.compute(
        predictions=preds,
        references=test_data["output"].to_list(),
        tokenizer=tokenizer.tokenize, rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    scores = rouge.compute(
        predictions=preds,
        references=test_data["output"].to_list(),
        tokenizer=tokenizer.tokenize,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_aggregator=False
    )
    # {"rouge1": [], "rouge2": [], "rougeL": []}
    
    score_names = scores.keys()
    
    for i in range(len(result_list)):
        a_score = {}
        for sname in score_names:
            a_score[sname] = scores[sname][i]
        result_list[i]["score"] = a_score

    result_json = {"total_score": total_score, "results": result_list}
    return result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--format", type=str, required=True, help="[my, koalpaca]")
    parser.add_argument("--test_file_path", type=str, required=True, help="[.csv]")
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()
    print(args)

    now = datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M')

    df = pd.read_csv(args.test_file_path)

    # koalpaca 방식
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    result_json = inference(model, tokenizer, df)
    
    dir_name = "test_result"
    file_name = args.name + "_" + args.model.strip().split("/")[-1] + "_"+ now +".json"
    
    os.makedirs(dir_name, exist_ok=True)
    
    path = os.path.join(dir_name, file_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False)

    print("Total ROUGE score:", result_json["total_score"])
    print("File saved at: ", path)
