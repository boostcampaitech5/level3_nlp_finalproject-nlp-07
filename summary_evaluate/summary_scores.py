import torch
from typing import List
from collections import defaultdict
import evaluate  # need to install rouge-score

from utils.preprocess import remove_tag, split_into_list


def fbeta_score(precision: float, recall: float, beta: float = 2.0):
    return (
        (1 + beta * beta)
        * precision
        * recall
        / (beta * beta * precision + recall)
        * 100
    )


def get_keyword_score(pred_keywords: List[List[str]], ref_keywords: List[List[str]]):
    """키워드 기반 점수 반환 [0, 100]
    1. `f2`: 개수를 고려해서 모든 키워드에 대한 f2 점수
    2. `rouge1`, `rouge2`, `rougeL`,`rougeLsum`
    """

    # 전체 요약문 기준 키워드 비교
    pred_all_keywords = defaultdict(lambda: 0)
    ref_all_keywords = defaultdict(lambda: 0)
    for pks in pred_keywords:
        for p in pks:
            pred_all_keywords[p] += 1
    for rks in ref_keywords:
        for r in rks:
            ref_all_keywords[r] += 1

    total_tp = 0
    total_fn = 0
    total_fp = 0

    for pk in pred_all_keywords:
        tp = min(pred_all_keywords[pk], ref_all_keywords[pk])
        fp = pred_all_keywords[pk] - tp
        total_tp += tp
        total_fp += fp
    for rk in ref_all_keywords:
        fn = max(0, ref_all_keywords[rk] - pred_all_keywords[rk])
        total_fn += fn

    recall = total_tp / (total_tp + total_fn)
    precision = total_tp / (total_tp + total_fp)

    # Summary-level Rouge-L 구하기
    pred_keyword_strings = [" ".join(pks) for pks in pred_keywords]
    ref_keyword_strings = [" ".join(rks) for rks in ref_keywords]
    pred_string = "\n".join(pred_keyword_strings)
    ref_string = "\n".join(ref_keyword_strings)

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=[pred_string],
        references=[ref_string],
        tokenizer=lambda x: x.split(),
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_aggregator=False,
    )
    rouge_scores = {s: rouge_scores[s][0] * 100 for s in rouge_scores}

    return {"f2": fbeta_score(precision, recall, 2.0), **rouge_scores}


def get_length_penalty(pred, ref):
    """반환: 길이 페널티, 길이 차이"""
    diff = len(pred) - len(ref)
    penalty = 0.0
    if diff > 0:
        diff = min(diff, len(ref))
        penalty = diff / len(ref)
    return penalty * 100, diff


def get_similarity(a, b):
    """두 문장의 임베딩을 받아 코사인 유사도 x 100 반환"""
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)).item() * 100


def get_sts_score(pred, ref, model, tokenizer):
    pred = remove_tag(pred)
    ref = remove_tag(ref)
    pred_list = split_into_list(pred)
    ref_list = split_into_list(ref)
    sentences = pred_list + ref_list
    pred_end = len(pred_list)  # sentence[:pred_end] 가 pred_list

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    embeddings, _ = model(**inputs, return_dict=False)

    score = 0.0

    # 각 ref 문장을 모든 pred 문장과 비교한 후, 최대 유사도의 평균을 점수로 사용
    for s in range(len(ref_list)):
        max_score = 0.0
        s += pred_end

        # ref 문장과 유사도가 가장 높은 pred 문장의 찾기
        for p in range(len(pred_list)):
            sim_score = get_similarity(embeddings[p][0], embeddings[s][0])
            if sim_score > max_score:
                max_score = max(max_score, sim_score)

        score += max_score

    final_score = score / len(ref_list)

    return final_score


if __name__ == "__main__":
    pred_keywords = [["가", "나", "라", "가"], ["가", "라"]]
    ref_keywords = [["가", "라", "가"], ["나", "나"]]

    print(get_keyword_score(pred_keywords, ref_keywords))
