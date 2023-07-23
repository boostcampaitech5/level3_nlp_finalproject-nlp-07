from typing import Union, List
import re
import os
import errno
from datetime import datetime
import json
from pytz import timezone


def save_evaluation(
    evaluation: dict,
    dir_name: str = "test_result",
    name: str = "",
    now: str = "",
) -> None:
    """`evaluation`을 json 파일에 저장

    파일 경로: {dir_name}/{name}_{model_name}_{now}.json

    Args:
        evaluation (dict): `evaluate()` 반환 값.\n
        dir_name (str, optional): 파일 저장 디렉토리 이름.\n
        name (str, optional): 파일 이름.\n
        model_name (str, optional): 사용한 모델 경로.\n
        now (str, optional): 평가 결과가 생성된 시각. 주어지지 않으면 함수 호출 당시 시각 사용.\n
    """
    if len(now) == 0:
        now = datetime.now(timezone("Asia/Seoul")).strftime("%m%d%H%M")

    file_name = ""
    if len(name) > 0:
        file_name += name + "_"
    file_name += now + ".json"

    os.makedirs(dir_name, exist_ok=True)

    path = os.path.join(dir_name, file_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False)

    print("File saved at: ", path)
