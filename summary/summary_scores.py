import torch
from typing import List
from collections import defaultdict
import evaluate

from utils.preprocess import remove_tag, split_into_list


def fbeta_score(precision: float, recall: float, beta: float = 2.0):
    return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)


def get_keyword_score(pred_keywords: List[List[str]], ref_keywords: List[List[str]]):
    """키워드 기반 점수 반환
    1. `f2`: 요약문 전체를 기준으로 계산한 f2 점수 x 100
    2. `rougeLsum`: Summary-level Rouge-L x 100
    """

    # 전체 요약문 기준 키워드 비교
    pred_all_keywords = defaultdict(int)
    ref_all_keywords = defaultdict(int)
    for pks in pred_keywords:
        for p in pks:
            pred_all_keywords[p] += 1
    for rks in ref_keywords:
        for r in rks:
            ref_all_keywords[r] += 1

    TP = 0
    FN = 0
    FP = 0

    for pk in pred_all_keywords:
        if ref_all_keywords[pk] > 0:
            TP += 1
            ref_all_keywords[pk] -= 1
        else:
            FP += 1
    for rk in ref_all_keywords:
        if pred_all_keywords[rk] > 0:
            pred_all_keywords[rk] -= 1
        else:
            FN += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    rouge = evaluate.load("rouge")
    pred_string = "\n".join([" ".join(pks) for pks in pred_keywords])
    ref_string = "\n".join([" ".join(rks) for rks in ref_keywords])
    rouge_score = rouge.compute(
        predictions=[pred_string],
        references=[ref_string],
        tokenizer=lambda x: x.split(),
        rouge_types=["rougeLsum"],
        use_aggregator=False,
    )["rougeLsum"][0]

    return {
        "f2": fbeta_score(precision, recall, 2.0) * 100,
        "rougeLsum": rouge_score * 100,
    }


# def get_f2_score(text:str, keywords: List[List[str]]):
#     """문장 별로 키워드 비교"""

#     text = remove_tag(text)
#     text_list = split_into_list(text)
#     score = 0
#     delim = ".+"
#     count = [0 for _ in text_list]
#     log = ""

#     for keyword in keywords:
#         log += f"Keyword {keyword}\n"
#         pattern = ""
#         for k in keyword.split():
#             pattern += k + delim
#         pattern = pattern[:-2] # 마지막 delim 제거
#         found = False
#         for tidx, t in enumerate(text_list):
#             res = re.search(pattern, t)
#             if res:
#                 log += f":: {t[:res.start()]}<{t[res.start():res.end()]}>{t[res.end():]}\n"
#                 count[tidx] += 1
#                 found = True
#         if found: score += 1

#     not_needed_text = [text_list[tidx] for tidx in range(len(text_list)) if count[tidx] == 0 ]

#     log += (
#         "===\n"
#         f"Has {score}/{len(keywords)} keywords.\n"
#         f"Has {len(not_needed_text)}/{len(text_list)} wrong lines.\n"
#     ) + str(not_needed_text)

#     keyword_score = score/len(keywords) # recall
#     line_score = (len(text_list) - len(not_needed_text)) / len(text_list) # precison
#     beta = 2 # beta times as much importance to recall as precision"
#     fbeta_score = (1 + beta*beta) * keyword_score * line_score / (beta*beta*line_score + keyword_score)

#     return fbeta_score * 100, log


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
