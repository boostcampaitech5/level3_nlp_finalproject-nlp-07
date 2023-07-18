# get preds from model -> summary_inference
# get evaluation scores
# save to file -> evaluate_scores.json, evaluate_results.json
from typing import List

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from summary_scores import get_f2_score, get_length_penalty, get_sts_score
from summary_utils import save_evaluation
from summary_inference import inference


MODEL = None
# model, tokenizer = load_model(MODEL)

# preds, times = summary_inference(model, tokenizer, dataset)

"""
preds = List[str]
times = List[ms]
"""

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
    test_dataset = [
        {"id": 1,
        "review":"가격도 싸고 월드콘 구구콘 두 가지 콘 아이스크림 먹을 수 있어 재구매 하였습니다. 맛도 달콤하고 예전부터 있던 상품으로 변함없는 맛이 좋습니다. 구입 기준 제조일자는 3주 전입니다. 두 가지 맛을 동시에 가격은 더 낮게 구입할 수 있어 만족하며 달콤하고 씹는 식감도 좋아 가족 모두가 잘 먹었습니다. 구입 기준 제조일자는 2주 전입니다. 맛은 짱 잘못 사면 콘이 누그러져서 맛없는데 와 이건 완전 바삭바삭 다 먹고 재구매해야겠네요. 최근에 만든 거 보내주셨네요. 오랜만에 아이스크림 주문했어요. 월드콘은 항상 맛있고 구구콘은 달콤한 게 댕길 때 더 맛남 평상시엔 좀 달게 느껴지는데  맛있게 잘 먹을게요. 부드럽고 달콤한 맛 뽀도독 씹히는 견과류의 고소한 맛은 옛날 어릴 적 추억여행을 하게 해줍니다. 언제 먹어도 맛난 아이스크림 진짜 맛있어요 더운 여름에 없어서는 안될 아이스크림콘도 바삭바삭 너무 많이네요  몇 번째 여기서만 주문해서 콘이 바삭바삭 너무 맛있어요. 여름이라 집에 항상 쟁여놓고 있네요. 전에 한번 구매해서 맛 좋아서 재구매합니다. 땡땡 잘 얼어서 배달됐네요. 콘도 바삭바삭 녹지 않고 맛있어요. 아이가 갑자기 아이스크림을 찾길래 주문해 봤어요. 하나도 안 녹고 진짜 잘 도착했어요. 월드콘이 줄었어요. 같이 온 구구는 그대로인데 월드콘 양이 많이 줄었네요. 얼었다 녹은 것도 아니고 월드콘만 다 저 크기예요 마트보다 쿠팡이 쌀 때가 많은데 수량이 많아 못 샀어요. 이번엔 반반씩 나와서 바로 샀어요. 여름엔 다양하게 사두는 편인데 쿠팡은 가격만 좋고 구성은 마트가 더 좋아요 큼직한 드라이아이스 3개로 밀봉포장되옴. 다 녹아오지 않을까 괜한 걱정이었음. 프레시 인정 많이 녹아 있었어요. 그래도 월드콘과 구구콘은 맛있지만요 오랫동안 먹어왔던 콘 아이스크림 집에서 받아서 먹으니 더 맛있다. 월드콘이랑 헷갈렸었나  맛있음. 근데 좀 녹아서 와서 까면 윗부분에 초코가 뚜껑에 붙어서 떼짐. 구구콘과 월드콘의 조합 좋아유 정말 좋아요. 만족도 백퍼입니다 10시 되기 전 주문하면 오후에 받을 수 있어서 좋았어요",
        "summary":"<가격> 싸고 <맛> 달콤하고 변함없는 맛 <맛> 두 가지 맛을 동시에 <가격> 더 낮게 <맛> 달콤하고 <식감> 씹는 식감도 좋아 <맛>완전 바삭바삭 부드럽고 달콤한 맛 뽀도독 씹히는 견과류의 고소한 맛 옛날 어릴 적 추억여행을 하게 해줍니다 <양> 월드콘 양이 많이 줄었네요",
        "keywords": ["싸", "달콤", "두 맛", "씹", "바삭", "부드럽", "고소", "양 줄"]
        },{
            "id": 2,
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

