import re

def preprocess(text):

    # \n \t 제거
    text = re.sub('[\s]', " ", text)

    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s0-9a-zA-Z\.]", " ", text)

    text = " ".join(text.split())


    # HTML 태그 제거
    text = re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", text).strip()

    # HTML 태그 내의 온점을 제외한 모든 문자 제거
    text = re.sub(r"<[^>.]+>\s+(?=<)|<[^>.]+>", "", text).strip()


    # 이메일 제거
    text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text).strip()

    # 해쉬태그 제거
    text = re.sub(r"#\S+", "", text).strip()

    # 멘션 제거
    text = re.sub(r"@\w+", "", text).strip()

    # URL 제거
    text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text).strip()
    text = re.sub(r"pic\.(\w+\.)+\S*", "", text).strip()

    # 특수 문자 제거
    text = re.sub(r"[^\w\s\.…]", "", text)

    # 문제를 일으킬 수 있는 문자 제거
    bad_chars = {"\u200b": "", "…": " ... ", "\ufeff": ""}
    for bad_char in bad_chars:
        text = text.replace(bad_char, bad_chars[bad_char])
    text = re.sub(r"[\+á?\xc3\xa1]", "", text)

    # 쓸모없는 괄호 제거
    bracket_pattern = re.compile(r"\((.*?)\)")
    modi_text = ""
    text = text.replace("()", "")  # 수학() -> 수학
    brackets = bracket_pattern.search(text)
    if brackets:
        replace_brackets = {}
        # key: 원본 문장에서 고쳐야하는 index, value: 고쳐져야 하는 값
        # e.g. {'2,8': '(數學)','34,37': ''}
        while brackets:
            index_key = str(brackets.start()) + "," + str(brackets.end())
            bracket = text[brackets.start() + 1: brackets.end() - 1]
            infos = bracket.split(",")
            modi_infos = []
            for info in infos:
                info = info.strip()
                if len(info) > 0:
                    modi_infos.append(info)
            if len(modi_infos) > 0:
                replace_brackets[index_key] = "(" + ", ".join(modi_infos) + ")"
            else:
                replace_brackets[index_key] = ""
            brackets = bracket_pattern.search(text, brackets.start() + 1)
        end_index = 0
        for index_key in replace_brackets.keys():
            start_index = int(index_key.split(",")[0])
            modi_text += text[end_index:start_index]
            modi_text += replace_brackets[index_key]
            end_index = int(index_key.split(",")[1])
        modi_text += text[end_index:]
        modi_text = modi_text.strip()
        text = modi_text

    # 기호 치환
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])
    text = text.strip()
    #
    # # 연속 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    return text


