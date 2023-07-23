from typing import List
from kiwipiepy import Kiwi, Token
from kiwipiepy.utils import Stopwords

from utils.preprocess import clean_text, split_into_list


class KeywordExtractor:
    """키워드를 추출하는 클래스. `get_keywords()` 사용."""

    def __init__(self):
        self.kiwi = Kiwi(model_type="sbg", typos="basic")

    def _get_stopwords(self, prod_name: str = "") -> Stopwords:
        stopwords = Stopwords()
        # 상품명에 있는 단어들 불용어 처리
        prod_name = clean_text(prod_name)
        if len(prod_name) > 0:
            for p in prod_name.split():
                self.kiwi.add_user_word(p)
                stopwords.add(p)

        stopwords.add(("배달", "NNG"))  # 일반명사
        stopwords.add(("구매", "NNG"))  # 일반명사
        stopwords.add(("냉동", "NNG"))  # 일반명사
        stopwords.add(("냉동식품", "NNG"))  # 일반명사
        stopwords.add(("주문", "NNG"))  # 일반명사
        stopwords.add(("거", "NNB"))  # 의존명사

        stopwords.add(("쿠팡", "NNP"))  # 고유명사
        stopwords.add(("먹", "VV"))
        # stopwords.add(("맛", "NNG")) # 일반명사
        # stopwords.add(("식감", "NNG")) # 일반명사
        # stopwords.add(("양", "NNG")) # 일반명사
        stopwords.add(("짱", "NNG"))  # 일반명사
        stopwords.add(("좋", "VA"))  # 일반명사
        stopwords.add(("같", "VA"))  # 일반명사
        stopwords.add(("받", "VV"))
        stopwords.add(("있", "VA"))

        stopwords.remove(("없", "VA"))
        stopwords.remove(("한", "MM"))

        return stopwords

    def _filter_tokens(self, token_list: List[Token]) -> List[str]:
        key_toks = dict()
        for tok in token_list:
            if (
                tok.tag.startswith("NNG")  # 일반 명사
                or tok.tag.startswith("NNP")  # 고유 명사
                or tok.tag.startswith("NR")  # 수사: 몇<백> 원
                or tok.tag.startswith("XR")  # 어근
                or tok.tag.startswith("VA")  # 형용사
                or tok.tag.startswith("VV")  # 동사
                or tok.tag.startswith("VCN")  # 부정 지시자(아니다)
                or tok.tag.startswith("SL")  # 알파벳
                or tok.tag.startswith("SN")  # 숫자
                or tok.tag.startswith("MM")  # 관형사
                or tok.form == "별로"  # 식감이 별로다
                or tok.form == "않"  # 신선하지 않다
                or tok.form.startswith("바삭")  # 바삭바삭 MAG
                or (tok.form == "안" and tok.tag == "MAG")  # 안 신선하다
                or (tok.form == "못" and tok.tag == "MAG")  # 안 신선하다
            ):
                key_toks[tok.form] = tok.tag

        return list(key_toks.keys())

    def get_keywords(self, text: str, prod_name: str = "") -> List[List[str]]:
        """
        Args:
            text (str): 키워드를 추출할 텍스트\n
            prod_name (str, optional): 상품명. 각 단어는 불용어 처리가 된다.\n
        """
        stopwords = self._get_stopwords(prod_name)

        sents = split_into_list(text)
        tokens_list = self.kiwi.tokenize(
            sents, normalize_coda=True, split_complex=True, stopwords=stopwords
        )

        keywords = []
        for tokens in tokens_list:
            tokens = self._filter_tokens(tokens)
            keywords.append(tokens)

        return keywords
