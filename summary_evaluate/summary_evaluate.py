from typing import List
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import gc
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from summary_scores import get_keyword_score, get_length_penalty, get_sts_score
from keyword_extractor import KeywordExtractor
from summary_utils import save_evaluation


def evaluate(
    preds: List[str], dataset, sts_model, sts_tokenizer, times: List[float] = None
):
    # dataset: [{id, review, summary, keywords}] 형태
    # preds는 dataset 순서대로라 가정

    # 컬럼 명
    ID = "id"
    REVIEW = "review"
    SUMMARY = "summary"
    PROD_NAME = "prod_name"

    assert (
        dataset[0].get(ID) and dataset[0].get(REVIEW) and dataset[0].get(SUMMARY)
    ), "dataset에 필요한 키 값이 들어있지 않습니다. {id, review, summary}"

    # 상품명 있는지 확인 (없어도 가능)
    has_prod_name = False
    if dataset[0].get(PROD_NAME):
        has_prod_name = True

    # 소요 시간 리스트 있는지 확인 (없으면 -1.0으로 초기화)
    if not times:
        times = [-1.0 for _ in preds]

    total_time = 0.0  # 리뷰 별 시간 평균
    total_keyword_scores = defaultdict(lambda: 0.0)
    total_sts_score = 0.0

    pred_len = len(preds)

    extractor = KeywordExtractor()

    results = []

    for pred, time, ref in tqdm(
        zip(preds, times, dataset), desc="Evaluating", total=pred_len
    ):
        if has_prod_name:
            prod_name = ref[PROD_NAME]
        else:
            prod_name = ""

        pred_keywords = extractor.get_keywords(pred, prod_name)
        ref_keywords = extractor.get_keywords(ref[SUMMARY], prod_name)

        keyword_scores = get_keyword_score(pred_keywords, ref_keywords)
        sts_score = get_sts_score(pred, ref[SUMMARY], sts_model, sts_tokenizer)
        _, length_diff = get_length_penalty(pred, ref[SUMMARY])

        item = {
            "id": ref[ID],
            "review": ref[REVIEW],
            "ref": ref[SUMMARY],
            "pred": pred,
            "time": time,
            "ref_keywords": ref_keywords,
            "pred_keywords": pred_keywords,
            **keyword_scores,
            "sts_score": sts_score,
            "length_diff": length_diff,
        }

        results.append(item)

        total_time += time
        total_sts_score += sts_score
        for score_name, score_val in keyword_scores.items():
            total_keyword_scores[score_name] += score_val

    for score_name, score_val in total_keyword_scores.items():
        total_keyword_scores[score_name] /= pred_len

    evaluation = {
        "total": {
            "time": total_time / pred_len,
            **total_keyword_scores,
            "sts_score": total_sts_score / pred_len,
        },
        "results": results,
    }
    for score_name, score_val in total_keyword_scores.items():
        evaluation["total"][score_name] = score_val

    return evaluation


if __name__ == "__main__":
    test_dataset = []  # 테스트 데이터셋
    preds, times = [], []  # 요약 모델로 생성한 요약문 리스트, 소요 시간 리스트

    import json
    import os

    READY_PATH = "/opt/ml/input/output/outout_T5_g256_ready.json"

    print("Read file:", READY_PATH)

    with open(READY_PATH, "r", encoding="utf-8") as f:
        ready_dict = json.load(f)
        test_dataset = ready_dict["test_dataset"]
        preds = ready_dict["preds"]
        times = ready_dict["times"]
        res_path = ready_dict["path"]

    dirname, filename = os.path.split(res_path)
    filename, ext = os.path.splitext(filename)
    filename += "_scores"

    ############ 평가 예시: 사용시 주석 처리 #############

    # from summary.summary_inference_v1 import get_review_summary

    # test_dataset = [
    #     {
    #         "id": 1,
    #         "prod_name": "떡볶이 추억의 국민학교 떡볶이 오리지널 (냉동), 600g, 2개",
    #         "review": "냄비에 물 360ml 받아 빨강 소스 한봉을 다. 깜장 소스는 단맛을 조절하는 소스예요. 떡이 쫀득하니 맛있네요. 어묵이 3장 들어있어요. 집에서 떡볶이 먹고 싶을 때 요. 밀키트로 만들어 먹기 좋아요.",
    #         "summary": "<조리> 냄비 <맛> 단맛 조절 <식감> 떡이 쫀득 <구성> 어묵 3장",
    #     }
    # ]

    # MODEL = "boostcamp-5th-nlp07/koalpaca-polyglot-5.8b-summary-v1.0"

    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL,
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    # ).to(device=f"cuda", non_blocking=True)

    # tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # for item in test_dataset:
    #     pred_list, time_list = get_review_summary(item["review"])
    #     preds.append(" ".join(pred_list))
    #     times.append(sum(time_list) / len(time_list))

    # del model
    # del tokenizer
    # gc.collect()

    ############ 평가 예시 끝 #############

    SENT_MODEL = "BM-K/KoSimCSE-roberta"
    print(f"Loading model and tokenizer for sts_score: {SENT_MODEL}")
    sentence_model = AutoModel.from_pretrained(SENT_MODEL)
    sentence_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
    eval_result = evaluate(
        preds, test_dataset, sentence_model, sentence_tokenizer, times
    )

    print("\n=== Total evaluation result ===")
    print(eval_result["total"])

    save_evaluation(eval_result, dir_name=dirname, name=filename)
