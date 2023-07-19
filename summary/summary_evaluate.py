from typing import List

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from summary_scores import get_f2_score, get_length_penalty, get_sts_score
from summary_utils import save_evaluation
from summary_inference import inference

def evaluate(preds: List[str], times: List[float], dataset, sts_model, sts_tokenizer):
    # dataset: [{id, review, summary, keywords}] 형태
    # preds는 dataset 순서대로라 가정
    
    # 컬럼 명
    ID = "id"
    KEYWORDS = "keywords"
    REVIEW = "review"
    SUMMARY = "summary"

    
    total_time = 0.0 # 리뷰 별 시간 평균
    total_f2_penalty = 0.0
    total_f2 = 0.0
    total_sts_score = 0.0
    
    LENGTH_PENALTY_WEIGHT = 0.1
    pred_len = len(preds)
    
    results = []
    
    for pred, time, ref in tqdm(zip(preds, times, dataset), desc="Evaluating", total=pred_len):
        f2, log = get_f2_score(pred, ref[KEYWORDS])
        sts_score = get_sts_score(pred, ref[SUMMARY], sts_model, sts_tokenizer)
        penalty, length_diff = get_length_penalty(pred, ref[SUMMARY])
        f2_penalty = (f2 - LENGTH_PENALTY_WEIGHT * penalty)
        
        item = {
            "id": ref[ID],
            "f2_penalty": f2_penalty,
            "f2": f2,
            "time": time,
            "sts_score": sts_score,
            "length_diff": length_diff,
            "log": log,
            "input": ref[REVIEW],
            "output": pred
        }
        results.append(item)
        
        total_f2_penalty += f2_penalty
        total_f2 += f2
        total_time += time
        total_sts_score += sts_score
    
    evaluation = {
        "total": {
            "f2_penalty": total_f2_penalty / pred_len,
            "f2": total_f2 / pred_len,
            "time": total_time / pred_len,
            "sts_score": total_sts_score / pred_len
        },
        "results": results
    }
    
    return evaluation


if __name__ == "__main__":
    test_dataset = None
    
    test_dataset = [{
            "id": 1,
            "review": "내 돈 내산 구매후기 2023년 4월 20일 떡볶이 간편하게 조리가 가능한 제품 냉동실에서 꺼낸 후 떡을 찬물이 5분 정도 담가 해동해 주세요. 냄비에 물 360ml 받아 빨강 소스 한봉을 다. 넣어줍니다. 깜장 소스는 단맛을 조절하는 소스예요. 너무 많이 넣으면 달 수 있어요. 1 3 정도 넣고 끓이라고 조리법에 나와있어요. 저는 너무 단 걸 싫어해서 쪼금만 넣었어요. 조리법대로 떡볶이만 넣고 만들어 드세요. 저는 쫄면 사리도 넣어서 먹고 싶어서 욕심부렸더니 처음 맛봤던 떡볶이 맛이랑 달라져서 아쉬웠어요. 저처럼 너무 많이 추가하지는 마세요. 떡이 쫀득하니 맛있네요. 어묵이 3장 들어있어요. 양배추와 파정도만 추가해서 끓이면 더욱 맛있을 거예요. 집에서 떡볶이 먹고 싶을 때 요. 밀키트로 만들어 먹기 좋아요. 2 3인이 먹기 좋은 떡볶이 저의 후기가 도움이 되셨다면 도움 돼요. 눌러주세요. 감사합니다.",
            "summary": "<재료> 밀떡볶이 <조리> 떡을 찬물에 5분정도 담가 해동 <조리> 냄비에 끓이라고 <재료> 깜장소스는 단맛을 조절하는 소스 <재료> 어묵 3장 <양> 2 3인이 먹기 좋은",
            "keywords": ["밀떡볶이","찬물 5분 해동", "냄비 끓", "깜장소스 단","어묵 3", "2 3인"]
        }
    ]
    
    
    MODEL = "boostcamp-5th-nlp07/koalpaca-polyglot-5.8b-summary-v1.0"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    preds, times = inference(model, tokenizer, test_dataset, prompt_template_name="v1.0")
    
    del model
    del tokenizer
    gc.collect()

    STS_MODEL = 'BM-K/KoSimCSE-roberta'
    print(f"Loading model and tokenizer for sts_score: {STS_MODEL}")
    sts_model = AutoModel.from_pretrained(STS_MODEL)
    sts_tokenizer = AutoTokenizer.from_pretrained(STS_MODEL)
    eval_result = evaluate(preds, times, test_dataset, sts_model, sts_tokenizer)
    
    print("\n=== Total evaluation result ===")
    print(eval_result["total"])
    print("\n=== Test result example ===")
    print(eval_result["results"][0])
    
    save_evaluation(eval_result)

