import re
import torch

from summary_utils import remove_tag, split_into_list


def get_f2_score(text, keywords):
    text = remove_tag(text)
    text_list = split_into_list(text)
    score = 0
    delim = ".+"
    count = [0 for _ in text_list]
    log = ""
    
    for keyword in keywords:
        log += f"Keyword {keyword}\n"
        pattern = ""
        for k in keyword.split():
            pattern += k + delim
        pattern = pattern[:-2] # 마지막 delim 제거
        found = False
        for tidx, t in enumerate(text_list):
            res = re.search(pattern, t)
            if res:
                log += f":: {t[:res.start()]}<{t[res.start():res.end()]}>{t[res.end():]}\n"
                count[tidx] += 1
                found = True
        if found: score += 1
    
    not_needed_text = [text_list[tidx] for tidx in range(len(text_list)) if count[tidx] == 0 ]
    
    log += (
        "===\n"
        f"Has {score}/{len(keywords)} keywords.\n"
        f"Has {len(not_needed_text)}/{len(text_list)} wrong lines.\n"
    ) + str(not_needed_text)
    
    keyword_score = score/len(keywords) # recall
    line_score = (len(text_list) - len(not_needed_text)) / len(text_list) # precison
    beta = 2 # beta times as much importance to recall as precision"
    fbeta_score = (1 + beta*beta) * keyword_score * line_score / (beta*beta*line_score + keyword_score)
    
    return fbeta_score * 100, log

def get_length_penalty(pred, ref):
    diff = len(pred) - len(ref)
    penalty = 0
    if diff > 0:
        diff = min(diff, len(ref))
        penalty = diff / len(ref)
    return penalty * 100, diff

def get_similarity(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)).item() * 100

def get_sts_score(pred, ref, model, tokenizer):
    pred = remove_tag(pred)
    ref = remove_tag(ref)
    pred_list = split_into_list(pred)
    ref_list = split_into_list(ref)
    sentences = pred_list + ref_list
    pred_end = len(pred_list) # sentence[:pred_end] 가 pred_list
    
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    embeddings, _ = model(**inputs, return_dict=False)
    
    score = 0.0
    
    for p in range(len(pred_list)):
        max_score = 0.0
        for s in range(len(ref_list)):
            s += pred_end
            sim_score = get_similarity(embeddings[p][0], embeddings[s][0])
            if sim_score > max_score:
                max_score = max(max_score, sim_score)
                
        score += max_score
            
    final_score = score / len(pred_list)
    
    return final_score

if __name__ == "__main__":
    print(get_f2_score("<시작> 가나마다 <두> 라마", ["가 마"]))