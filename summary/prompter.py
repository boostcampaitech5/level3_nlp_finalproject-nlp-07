import os
import json
from typing import Union


class Prompter(object):
    def __init__(self, template_name: str = ""):
        if not template_name:
            template_name = "v1.0"
            
        prompt_file_path = f"./templates/{template_name}.json"
        if not os.path.exists(prompt_file_path):
            raise ValueError(f"Can't read {prompt_file_path}")
        with open(prompt_file_path, "r") as f:
            self.template = json.load(f)["template"]
         

    def generate_prompt(
        self,
        search_name: Union[None, str] = None,        
        review: Union[None, str] = None,
        summary: Union[None, str] = None,
    ) -> str:
        if search_name:
            res = self.template.format(
                search_name=search_name,
                review=review
            )
        else:
             res = self.template.format(
                review=review
            )
             
        if summary:
            res += " " + summary + "<|endoftext|>"
        
        return res

    
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