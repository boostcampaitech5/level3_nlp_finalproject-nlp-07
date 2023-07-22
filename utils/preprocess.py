from typing import List
import re

def clean_text(sent:str, remove_tag:bool = True, remove_dot = True)->str:
    """
    특수 문자, 문장 부호 제거. 
    
    - `remove_tag`: 특성 태그 제거 여부. 
    - `remove_dot`: 마침표 제거 여부. 제거하지 않을 때, 연속된 마침표는 한 개로 변환.
    """
    if remove_tag:
        sent = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s0-9a-zA-Z.]", " ", sent)
    else:
        sent = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s0-9a-zA-Z<>.]", " ", sent)
    
    if remove_dot:
        sent = re.sub("\.", " ", sent)
    else:
        sent = re.sub("\.{2,}", ".", sent)
     
    sent = re.sub("[ㄱ-ㅎㅏ-ㅣ]+", " ", sent) # 초성체 제거
    sent = " ".join(sent.split()) # 공백 최소화
    sent = sent.strip()
    return sent

def get_no_space_length(text):
    """공백 제외 글자수 반환"""
    return len(re.sub("[ \n]", "", text))

def remove_tag(text:str)->str:
    """'<>' 를 마침표로 변환
    
    eg) <특성> 내용 <특성2> 내용2 &rarr; 내용. 내용2
    """
    
    text = re.sub(" <[^>]+>", ".", text) # 태그 마침표로 변환
    text = re.sub("<[^>]+>", "", text) # 맨 앞 태그 제거
    text = " ".join(text.split())
    text = text.strip()
    return text

def split_into_list(text:str)->List[str]:
    """'<>' 를 구분자로 사용하여 `text`를 여러 문장으로 분리
    
    eg) <특성> 내용 <특성2> 내용2 &rarr; ["내용", "내용2"]
    """
    
    text = remove_tag(text)
    return [t.strip() for t in text.split(".")]