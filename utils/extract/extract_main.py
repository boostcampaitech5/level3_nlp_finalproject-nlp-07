from fastapi import FastAPI
import os
import openai
from pydantic import BaseModel
import configparser
from typing import List, Union, Optional, Dict, Any


app = FastAPI()

env = os.getenv('MY_APP_ENV', 'local')
config = configparser.ConfigParser()
config.read(f'../../config/config-{env}.ini')
openapi = config['openai']
openai.api_key=openai['api_key']

class Item(BaseModel):
    query: str

def gpt_chat_gen(prompt, model="gpt-3.5-turbo"):
    
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def build_text(input):
    """
    입력 프롬프트 생성
    input: 사용자로 부터 입력받은 문장
    """
    promprt = """
    답변 외에는 다른 출력을 하지마
    
    내가 문장을 입력하면, 문장에서 '상품명'과 그 상품에 대한 '조건'을 출력해줘
    '조건'은 명사로 출력해야돼
    
    아래는 '조건'에 대한 예시야
    1. 양이 적당한, 혼자 먹기 좋은 : 양
    2. 가격이 싼, 가성비가 좋은 : 가격
    3. 조리가 간편한, 쉽게 먹을 수 있는 : 간편함
    
    아래는 예시야
    1. 문장 : '맵지 않고 쫄깃한 떡볶이를 추천해줘', 답변 : { '상품' : '떡볶이', '조건' : ['맵기','식감'] }
    2. 문장 : '냉동삼겹살인데 보관이 편리하고 가성비가 좋은 상품을 추천해줘', 답변 : { '상품' : '냉동삼겹살', '조건' : ['보관', '가성비'] }
    3. 문장 : '신선하고 달고 가격이 적당한 귤을 추천해줘', 답변 : { '상품' : '귤', '조건' : ['신선함', '당도', '가성비'] }
    4. 문장 : '혼자 먹기 좋은 곱창전골을 추천해줘', 답변 : { '상품' : '곱창전골', '조건' : ['양'] }
    
    문장 : 
    """
    text = (promprt + input)

    return text

def extract(prompt):
    """
    추출결과 dictionary 형식으로 
    :param prompt: 프롬프트 형식을 지닌 입력 텍스트
    """
    result = gpt_chat_gen(prompt)
    return result

@app.post("/chuchul")
def get_extract(item: Item):
    input_promprt = build_text(item.query)
    result = extract(imput_promprt)
    return result