from typing import List
from fastapi import FastAPI
import re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

MODEL = "boostcamp-5th-nlp07/koalpaca-polyglot-5.8b-summary-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# 입력 프롬프트 제작
def build_text(review):
    """
    입력 프롬프트 생성
    :param review: 전처리된 입력 텍스트
    """
    text = (  # multi-line string
        f"### 명령어: 다음 상품에 대한 리뷰에서 상품의 특성을 요약하세요\n\n"
        "### 리뷰: " + review + "\n\n"
        "### 요약:"
    )

    return text

def ask(prompt):
    """
    모델 추론 결과 생성
    :param prompt: 프롬프트 형식을 지닌 입력 텍스트
    """

    ans = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    result = ans[0]["generated_text"]
    return result

def clean_text(text, type="pre") -> str:
    """
    텍스트 전처리 및 후처리    
    :param sent: 처리할 텍스트
    :param type: `pre` - 입력 텍스트 전처리, `post` - 생성 결과 후처리
    :return:
    """
    if type == "pre":
        text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s0-9a-zA-Z]", " ", text)
    elif type == "post":
        text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s0-9a-zA-Z<>]", " ", text)
    text = re.sub("[ㄱ-ㅎㅏ-ㅣ]+", " ", text)
    text = " ".join(text.split())
    text = text.strip()
    return text


def get_no_space_length(text):
    """공백 미포함 글자수 반환"""
    return len(re.sub("[ \n]", "", text))

def generate_summary(text):
    text = clean_text(text, type="pre")
    prompt = build_text(text)
    summary = ask(prompt)
    summary = clean_text(summary, type="post")
    return summary
    

@app.post("/summary")
def get_review_summary(reviews: List[str]):
    input_text = ""
    summary = ""
    
    for review in reviews:
        if get_no_space_length(input_text) + get_no_space_length(review) > 500:
            result = generate_summary(input_text)
            summary = summary + " " + result
            input_text = ""
        
        input_text = input_text + " " + review
            
    
    result = generate_summary(input_text)
    summary = summary + " " + result
    
    return summary.strip()