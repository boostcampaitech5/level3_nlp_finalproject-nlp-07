import os
import json
from typing import Union

class Prompter(object):
    def __init__(self, template_path: str = ""):
        
        self.template = None
        self.keys = None
        
        assert os.path.exists(template_path), f"Template path does not exist: {template_path}"
        
        with open(template_path, "r") as f:
            template_json = json.load(f)
            self.template = template_json["template"]
            self.keys = template_json["keys"]
         
    def generate_prompt(self, **kwargs):
        for k in self.keys:
            assert k in kwargs, f"Make sure to set all following parameters: {self.keys}"
        
        return self.template.format(**kwargs)

# 입력 프롬프트 제작 (미사용)
def build_text(search_name, review, template_name):
    if template_name == "my-v0":
        text = (  # multi-line string
            f"### 명령어: 다음의 {search_name}에 대한 리뷰를 특성 세부설명 형식으로 요약하세요\n\n"
            "### 리뷰: " + review.strip() + "\n\n"
            "### 요약:"
        )
    elif template_name == "my-v1":
        text = (
            f"### 명령어: 다음 상품에 대한 리뷰에서 상품의 특성을 요약하세요\n\n"
            "### 리뷰: " + review + "\n\n"
            "### 요약:"
        )
    elif template_name == "koalpaca":
        text = (
            f"### 질문: 다음의 {search_name}에 대한 리뷰를 특성 세부설명 형식으로 요약하세요.\n\n"
            "" + review.strip() + "\n\n"
            "### 답변:"
        )
    elif template_name == "kullm":
        instruction = f"다음의 {search_name}에 대한 리뷰를 특성 세부설명 형식으로 요약하세요."

        text = "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:\n".format(
            instruction=instruction, input=review
        )

    return text