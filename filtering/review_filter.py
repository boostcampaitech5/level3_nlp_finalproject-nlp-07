import re
from tqdm import tqdm
from typing import List, Union

from kiwipiepy import Kiwi

class ReviewFilter:
    def __init__(self):
        self.kiwi = Kiwi()
        self.del_patterns = [
            "조리", "해동", "요리", '추가','[센중약]불', "넣", "헹", # 레시피
            # "에어[ ]?[프후]라이[기어]?", "오븐", "전자레인지", "[^ ]*팬", "냄비",
            '작성', '도움', '내돈내산','리뷰','안녕','보답','감사','눌러','좋은 하루', "후기", # 리뷰 끝맺음
            "(유통)?기한", 
            "(재)?구매", "(재)?구입",
            "배달", "배송", "로켓", "프레시",'주문',
        ]
    
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
        
        if verbose:
            short_tqdm = tqdm(corpus, total=len(corpus), desc="Filtering too short sentences")
        else:
            short_tqdm = corpus
        
        filtered_corpus = [text for text in short_tqdm
                        if (
                            (include_blank and len(text) >= min_len)
                            or (not include_blank and len(re.sub(" ", "", text)) >= min_len)
                        )]
        short_len = len(corpus) - len(filtered_corpus)
        
        if verbose:
            long_tqdm = tqdm(filtered_corpus, total=len(filtered_corpus), desc="Filtering too long sentences")
        else:
            long_tqdm = filtered_corpus
        
        filtered_corpus = [text for text in long_tqdm 
                        if(
                            (include_blank and len(text) <= max_len)
                            or (not include_blank and len(re.sub(" ", "", text)) <= max_len)
                        )]
        long_len = len(corpus) - len(filtered_corpus)
        
        if verbose:
            print(f"Shorter than {min_len}: {short_len} | Longer than {max_len}: {long_len}")
            
        return filtered_corpus
    
    
    def filter_text(self, text, verbose=False, join = True)->Union[str, List[str]]:
        """`text`를 문장으로 분리하고, 문장 별 필터링
        
        - `join = True`: 분리한 문장들을 `". ".join()` 후 반환
        - `join = False`: 문장 리스트 반환
        """
        
        # 문장 분리
        corpus = []
        sents = self.kiwi.split_into_sents(
                text, 
                normalize_coda=True, 
                split_complex=True,
            )
        corpus = [se.text for se in sents]
        
        # 길이 필터링
        filtered_corpus = self.filter_length(corpus, 10, 200, verbose=verbose)
        
        # 정규식 패턴 필터링
        del_dict = self.check_patterns(self.del_patterns, filtered_corpus, verbose=verbose)
        
        if verbose:
            filtered_corpus_tqdm = tqdm(enumerate(filtered_corpus), total=len(filtered_corpus), desc="Filtering corpus")
        else:
            filtered_corpus_tqdm = enumerate(filtered_corpus)
        
        filtered_corpus = [
            re.sub("\.", "", text) 
            for idx, text in filtered_corpus_tqdm
            if not del_dict.get(idx)
            ]
        
        # 문장 붙여서 반환. 구분자로 마침표 사용
        if join:
            return ". ".join(filtered_corpus) + "."
        else:
            return filtered_corpus
    
if __name__ == "__main__":
    
    text = "곰곰 소중한 우리 쌀 2022년산 20kg 상 등급 1개 매번 주문하는 쌀입니다 어쩌다 다른 쌀을 주문해서 먹었는데 정말 실망하고 역시 곰곰 우리 쌀이 제일이라는 걸 느꼈네요.. 일단은 가격에 부담스럽지 않아서 주문하는데요 가격만 저렴하고 쌀이 정말 못 먹겠음 안 사잖아요 요건 안 그래요 가격도 착한데 쌀 품질도 좋아요 비싼 쌀은 6만 원대가 넘어가는데 다섯 식구인 저희는 그런 쌀은 못 먹겠더라고요.. ᅲᅲ 근데 요 곰곰 쌀이 정말 맛이 좋고 윤기도 흘러요 밥할 때 구수한 냄새 있잖아요 그게 쌀도 좋아요 좋은 냄새가 나더라고요 밥할 때마다 구수한 냄새가 입맛이 확 돋아나게 합니다. 배송도 꼼꼼하게 잘 오구요 집 앞까지 가져다주시니 정말 땡큐죠 쌀 깨진 것 거의 없고 부서짐이나 가루도 거의 없고요 저렴하게 좋은 쌀 드시고 싶으신 분들 꼭 드셔보세요 한번 빠지면 헤어 나올 수 없는 곰곰입니다"  
    
    filter = ReviewFilter()
    
    res = filter.filter_text(text, verbose=True, join=False)
    print(res)
    
    res = filter.filter_text(text, verbose=True)
    print(res)
    print(filter.get_filter_score(len(text), len(res), verbose=True))
        
        
        