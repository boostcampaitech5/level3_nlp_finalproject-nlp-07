from typing import Union


def clean_text(sent, remove_tag = True):
    """
    특수 문자, 문장 부호, 조건 태그 제거
    """
    if remove_tag:
        sent = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s0-9a-zA-Z]", " ", sent)
    else:
        sent = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s0-9a-zA-Z<>]", " ", sent)
    sent = re.sub("[ㄱ-ㅎㅏ-ㅣ]+", "", sent) # 초성체 제거
    sent = " ".join(sent.split()) # 공백 최소화
    sent = sent.strip()
    return sent

def remove_tag(text):
    text = re.sub(" <[^>]+>", ".", text) # 태그 마침표로 변환
    text = re.sub("<[^>]+>", "", text) # 맨 앞 태그 제거
    text = " ".join(text.split())
    text = text.strip()
    return text

def split_into_list(text):
    text = remove_tag(text)
    return [t.strip() for t in text.split(".")]
import re
import os
from datetime import datetime
import json
import torch
from pytz import timezone

def save_evaluation(evaluation, dir_name = "test_result", name = "", model_name = "", now = ""):
    if len(now) == 0:
        now = datetime.now(timezone("Asia/Seoul")).strftime("%m%d%H%M")
    
    file_name = ""
    if len(name) > 0:
        file_name += name + "_"
    if len(model_name) > 0:
        file_name += model_name.strip().split("/")[-1] + "_"
    file_name += now + ".json"

    os.makedirs(dir_name, exist_ok=True)

    path = os.path.join(dir_name, file_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False)

    print("File saved at: ", path)
