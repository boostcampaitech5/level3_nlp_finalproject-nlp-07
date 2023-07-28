import re
from tqdm import tqdm
from typing import List, Union
from collections import defaultdict

from kiwipiepy import Kiwi

class ReviewFilter:
    def __init__(self):
        self.kiwi = Kiwi()
        self.del_patterns = [
            "조리", "해동", "요리", '추가','[센중약]불', "넣", "헹", "방법", # 레시피
            # "에어[ ]?[프후]라이[기어]?", "오븐", "전자레인지", "[^ ]*팬", "냄비",
            '작성', '도움', '내돈내산','리뷰','안녕','보답','감사','눌러','좋은 하루', "후기", # 리뷰 끝맺음
            "(유통)?기한", "보관",
            "(재)?구매", "(재)?구입",
            "배달", "배송", "로켓", "프레시",'주문',
        ]
    
    def split_into_sents(self, text):
        sents = self.kiwi.split_into_sents(
                text, 
                normalize_coda=True, 
                split_complex=True,
            )
        return [se.text for se in sents]
    
    def get_filter_score(self, before_len:int, after_len:int, verbose=False):
        """필터링 비율 반환"""
        perc = (1 - after_len/before_len)*100
        
        if verbose:
            print(f"{before_len} -> {after_len}, 삭제 개수: {before_len - after_len}, 필터링 비율: {perc: .02f}%")
        return perc
    
    def check_patterns(self, patterns: List[str], corpus: List[str], highlight=True, verbose=False):
        """주어진 정규표현식에 해당하는 단어가 있는 문장 반환"""
        res_dict = dict()

        if verbose:
            corpus_tqdm = tqdm(enumerate(corpus), total=len(corpus), desc="Checking patterns")
        else:
            corpus_tqdm = enumerate(corpus)

        for idx, text in corpus_tqdm:
            to_save = False
            for k in patterns:
                res = re.search(k, text)
                if res:
                    to_save = True
                    if highlight: text = text[:res.start()] + "[" + res.group() + "]" + text[res.end():]
            if to_save: 
                res_dict[idx] = text

        return res_dict
    
    def filter_length(self, corpus, min_len=10, max_len=200, include_blank = False, verbose=False):
        """[min_len, max_len] 사이에 있는 문장 외 필터링"""
        
        filtered_corpus = []
        short_len = 0
        long_len = 0
        for text in corpus:
            if include_blank: 
                text_len = len(text)
            else:
                text_len = len(re.sub(" ", "", text))
                
            if text_len < min_len: 
                short_len += 1
            elif text_len > max_len:
                max_len += 1
            else:
                filtered_corpus.append(text)
        
        if verbose:
            print(f"Shorter than {min_len}: {short_len} | Longer than {max_len}: {long_len}")
            
        return filtered_corpus
    
    def filter_duplicates(self, corpus, verbose=False):
        """
        Returns:
            Tuple(List[str], List[int]): (중복 제거된 corpus, 제거된 문장 인덱스)
        """
        unique_corpus = defaultdict(int)
        remove_idx = []
        for idx, c in enumerate(corpus):
            unique_corpus[c] += 1
            if unique_corpus[c] > 1: remove_idx.append(idx)
        
        if verbose:
            print(f"Removed {len(remove_idx)} duplicate sentences.")
            
        return list(unique_corpus.keys()), remove_idx
            
    
    def filter_text(self, text, verbose=False, join = True)->Union[str, List[str]]:
        """`text`를 문장으로 분리하고, 문장 별 필터링
        
        - `join = True`: 분리한 문장들을 `". ".join()` 후 반환
        - `join = False`: 문장 리스트 반환
        """
        
        # 문장 분리
        corpus = self.split_into_sents(text)
        
        # 중복 필터링
        if verbose: print("Filtering duplicates.")
        filtered_corpus = self.filter_duplicates(corpus, verbose=verbose)[0]
        
        # 길이 필터링
        if verbose: print("Filtering by length.")
        filtered_corpus = self.filter_length(filtered_corpus, 10, 200, verbose=verbose)
        
        # 정규식 패턴 필터링
        del_dict = self.check_patterns(self.del_patterns, filtered_corpus, verbose=verbose)
        
        if verbose:
            filtered_corpus_tqdm = tqdm(enumerate(filtered_corpus), total=len(filtered_corpus), desc="Filtering corpus")
        else:
            filtered_corpus_tqdm = enumerate(filtered_corpus)
        
        filtered_corpus = [
            re.sub("\.", "", text) + "."
            for idx, text in filtered_corpus_tqdm
            if not del_dict.get(idx)
            ]
        
        # 문장 붙여서 반환. 구분자로 마침표 사용
        if join:
            return " ".join(filtered_corpus)
        else:
            return filtered_corpus
    
if __name__ == "__main__":
    
    # text = "떡볶이는 항상 냉동실에 한 개는 있어야 된다는 급당기면 바로 해먹기 편해요 오랜만에 국떡이가 당겨서 구매했어요 600그램 혼자 먹기 딱 좋은 양이에요 떡. 어묵 3장. 소스가 두 개 냉장실에 하루 빼놨더니 해동이 잘 됐어요 물 360미리 넣고 떡. 어묵 3개. 소스 한 번에 다 넣고 10분 정도 끓여주면 옛날 떡볶이가 완성 너무너무 간단해요 밀떡이 아주아주 쫄깃쫄깃 식감이 좋고 어묵 3개도 얇은 게 아니고 조금 도톰해서 더 맛있는 거 같아요 떡볶이 냄새가 죽여줘요 떡볶이 떡이 몇 시간이 지났는데도 딱딱해지지 않고 말랑말랑 식었는데도 맛나요 가격 대비 매우 만족합니다 유통기한도 엄청 길어서 냉동실에 쟁여놔도 좋겠어요 또 구매할게요" 
    text = "이것은 테스트입니다. 중복 문장 테스트 긴 문장으로. 하하하 중복 문장 테스트. 하하핫. 중복 문장 테스트 긴 문장으로. "
    
    import sys
    sys.path.append("/opt/ml/input/level3_nlp_finalproject-nlp-07")
    from summary.summary_utils import clean_text
    
    filter = ReviewFilter()
    
    text = clean_text(text, remove_dot=False) # 마침표는 문장 분리에 도움을 주기 때문에 제거하지 않음
    print("\n=== 전 ===")
    sents = [
        s.text for s in filter.kiwi.split_into_sents(
                text, 
                normalize_coda=True, 
                split_complex=True,
            )]
    print(sents)
    print("\n=== 후 ===")
    res = filter.filter_text(text, verbose=True, join=False)
    print(res)
    
    res = filter.filter_text(text, verbose=True)
    print()
    print(res)
    print("=== 필터링 비율 ===\n")
    print(filter.get_filter_score(len(text), len(res), verbose=True))
        
        
        