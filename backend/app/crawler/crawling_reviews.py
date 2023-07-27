from bs4 import BeautifulSoup as bs
from pathlib import Path
from typing import Optional,Union,Dict,List
from openpyxl import Workbook
import time
import os
import re
import requests as rq
import json
import csv
from datetime import datetime
from pytz import timezone
from tqdm import tqdm

import sys

# Set your project path
project_path = os.path.expanduser('~/level3_nlp_productserving-nlp-07/backend/crawler')

# Add the project path to the PYTHONPATH
sys.path.append(project_path)


from preprocessing.text_preprocessing import preprocess
from preprocessing.text_hanspell import spell_check
from pathlib import Path


def get_headers(
    key: str,
    default_value: Optional[str] = None
    )-> Dict[str,Dict[str,str]]:


    """ Get Headers """
    # JSON_FILE : str = './json/headers.json'
    JSON_FILE = Path(__file__).resolve().parent / 'json' / 'headers.json'

    with open(JSON_FILE,'r',encoding='UTF-8') as file:
        headers : Dict[str,Dict[str,str]] = json.loads(file.read())

    try :
        return headers[key]
    except:
        if default_value:
            return default_value
        raise EnvironmentError(f'Set the {key}')

class Coupang:

    @staticmethod
    def get_product_code(url: str)-> str:
        """ 입력받은 URL 주소의 PRODUCT CODE 추출하는 메소드 """
        prod_code : str = url.split('products/')[-1].split('?')[0]
        return prod_code

    def __init__(self, max_reviews_per_url: int)-> None:
        self.__headers : Dict[str,str] = get_headers(key='headers')
        self.MAX_REVIEWS_PER_URL = max_reviews_per_url


    def main(self, url_list: List[str], prod_names: List[str], search_names: List[str],  writer: csv.DictWriter) -> None:
        # 각 URL의 첫 페이지에서 리뷰를 가져옴
        result = []
        idx = 0
        for URL in tqdm(url_list):

            # URL의 Product Code 추출
            prod_code: str = self.get_product_code(url=URL)
            review_counter = 0  # 각 URL별로 카운터 초기화

            review_counter = 0

            prod_name = prod_names[idx]
            search_name = search_names[idx]
            idx += 1

            for page in range(1, 100):
                # URL 주소 재가공
                url_to_fetch = f'https://www.coupang.com/vp/product/reviews?productId={prod_code}&page={page}&size=10&sortBy=ORDER_SCORE_ASC&ratings=&q=&viRoleCode=3&ratingSummary=true'

                if review_counter >= self.MAX_REVIEWS_PER_URL:
                    break

                # __headers에 referer 키 추가
                self.__headers['referer'] = URL

                with rq.Session() as session:
                    page_data, review_counter = self.fetch(url=url_to_fetch, session=session, review_counter=review_counter, prod_name=prod_name, search_name=search_name)
                    # result.append(page_data)

                    # review_content spell_check 처리 and write to CSV
                    for review in page_data:
                        review['review_content'] = spell_check(review['review_content'])
                        writer.writerow(review)



        # review_content spell_check 처리
        for page_data in result:
            for review in page_data:
                review['review_content'] = spell_check(review['review_content'])

        return result

    def fetch(self,url:str,session, review_counter, prod_name:str, search_name:str)-> List[Dict[str,Union[str,int]]]:
        save_data : List[Dict[str,Union[str,int]]] = list()

        with session.get(url=url,headers=self.__headers) as response :
            html = response.text
            soup = bs(html,'html.parser')

            # Article Boxes
            article_lenth = len(soup.select('article.sdp-review__article__list'))

            # brand_name = soup.find_all('a', 'prod-brand-name')[0].text
            # print(brand_name)

            # # 브랜드명
            # brand_name = driver.find_element(By.CLASS_NAME, 'prod-sale-vendor-name').text

            # # 브랜드점수
            # brand_rating = driver.find_element(By.CLASS_NAME, 'seller-rating with-company').text


            for idx in range(article_lenth):

                # 이미 충분한 리뷰를 수집한 경우 루프 종료
                if review_counter >= self.MAX_REVIEWS_PER_URL:
                    break


                dict_data : Dict[str,Union[str,int]] = dict()
                articles = soup.select('article.sdp-review__article__list')

                # 구매자 이름
                user_name = articles[idx].select_one('span.sdp-review__article__list__info__user__name')
                if user_name == None or user_name.text == '':
                    user_name = '-'
                else:
                    user_name = user_name.text.strip()

                top100_yn = articles[idx].select_one('img.sdp-review__article__list__info__top-badge')
                if top100_yn == None or top100_yn.text == '':
                    top100_yn = 'N'
                else:
                    top100_yn = 'Y'
                # print(top100_yn)

                # 평점
                rating = articles[idx].select_one('div.sdp-review__article__list__info__product-info__star-orange')
                if rating == None:
                    rating = 0
                else :
                    rating = int(rating.attrs['data-rating'])

                # 구매자 상품명
                # prod_name = articles[idx].select_one('div.sdp-review__article__list__info__product-info__name')
                # if prod_name == None or prod_name.text == '':
                #     prod_name = '-'
                # else:
                #     prod_name = prod_name.text.strip()

                # 헤드라인(타이틀)
                headline = articles[idx].select_one('div.sdp-review__article__list__headline')
                if headline == None or headline.text == '':
                    headline = ''
                else:
                    headline = headline.text.strip()

                # 리뷰 내용
                review_content = articles[idx].select_one('div.sdp-review__article__list__review > div')
                if review_content is None:
                    review_content = ''
                else:
                    review_content = preprocess(review_content.text.strip())

                if len(review_content) < 50:
                    continue
                if len(review_content) > 500:
                    continue


                # 맛 만족도
                answer = articles[idx].select_one('span.sdp-review__article__list__survey__row__answer')
                if answer == None or answer.text == '':
                    answer = ''
                else:
                    answer = answer.text.strip()

                # divs = soup.find_all('div', class_='sdp-review__article__list__survey__row')
                # suryey_list = []
                # # 각 div 요소에 대한 정보를 출력합니다
                # for div in divs:
                #     question = div.find('span', class_='sdp-review__article__list__survey__row__question').text
                #     answer = div.find('span', class_='sdp-review__article__list__survey__row__answer').text
                #     # print(f"Question: {question}, Answer: {answer}")
                #     mix_text = f'<{question}>{answer}'
                #     suryey_list.append(mix_text)
                # survey = ''.join(suryey_list)
                # print(suryey)

                helped_cnt = articles[idx].select_one('.js_reviewArticleHelpfulContainer')
                if helped_cnt == None or helped_cnt.text == '':
                    helped_cnt = 0
                else:
                    help_cnt_str = helped_cnt.text.strip().split('명에게 도움 됨')[0]  # Split the string and get the first part
                    help_cnt_str = help_cnt_str.replace(',', '')  # Remove the comma
                    helped_cnt = int(help_cnt_str)  # Then convert it to integer
                # print(help_cnt)

                if helped_cnt < 1:
                    continue



                # 원본 URL
                dict_data['prod_name'] = prod_name
                dict_data['user_name'] = user_name
                dict_data['rating'] = rating
                dict_data['headline'] = headline
                dict_data['review_content'] = review_content
                dict_data['answer'] = answer
                dict_data['helped_cnt'] = helped_cnt
                # dict_data['survey'] = survey
                dict_data['top100_yn'] = top100_yn
                dict_data['search_name'] = search_name

                review_counter += 1

                save_data.append(dict_data)

                # print(dict_data , '\n')

                # Add delay
                # time.sleep(1)                 

            return save_data, review_counter

    def input_review_url(self)-> str:
        while True:
            # Window
            # os.system('cls')
            # Mac
            os.system('clear')
            
            # Review URL
            review_url : str = input('원하시는 상품의 URL 주소를 입력해주세요\n\nEx)\nhttps://www.coupang.com/vp/products/7335597976?itemId=18741704367&vendorItemId=85873964906&q=%ED%9E%98%EB%82%B4%EB%B0%94+%EC%B4%88%EC%BD%94+%EC%8A%A4%EB%8B%88%EC%BB%A4%EC%A6%88&itemsCount=36&searchId=0c5c84d537bc41d1885266961d853179&rank=2&isAddedCart=\n\n:')
            if not review_url :
                # Window
                os.system('cls')
                # Mac
                #os.system('clear')
                print('URL 주소가 입력되지 않았습니다')
                continue
            return review_url

    def input_page_count(self)-> int:
        # Window
        # os.system('cls')
        # Mac
        os.system('clear')
        while True:
            page_count : str = input('페이지 수를 입력하세요\n\n:')
            if not page_count:
                print('페이지 수가 입력되지 않았습니다\n')
                continue

            return int(page_count)
        
# def load_urls(file_path: str) -> List[str]:
#     # Load URLs from a CSV file
#     with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         urls = [row['URL'] for row in reader]
#         prod_names = [row['name'] for row in reader]
#         search_names = [row['search_name'] for row in reader]
#     return urls, prod_names, search_names

def load_urls(file_path: str) -> List[str]:
    urls = []
    prod_names = []
    search_names = []

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            urls.append(row['URL'])
            prod_names.append(row['name'])
            search_names.append(row['search_name'])

    return urls, prod_names, search_names

class CSV:
    @staticmethod
    def save_file(file_name, max_reviews_per_url)-> None:

        file_path = f"./{file_name}.csv"
        url_list, prod_names, search_names = load_urls(file_path)

        # 크롤링 결과
        # results : List[List[Dict[str,Union[str,int]]]] = Coupang().main(url_list, prod_names)

        # 파일에 쓸 데이터 준비
        csv_columns = ['prod_name', 'user_name', 'rating', 'headline', 'review_content', 'answer', 'helped_cnt', 'top100_yn', 'search_name']
        # 서울 시간
        return_file_name = f'review_{file_name}'

        # 파일 이름
        csv_file = f'./{return_file_name}.csv'

        # with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        #     writer.writeheader()
        #     for data in results:
        #         for item in data:
        #             writer.writerow(item)
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            # 크롤링 결과
            Coupang(max_reviews_per_url).main(url_list, prod_names, search_names, writer)        

        print(f'파일 저장완료!\n\n{csv_file}')

        return return_file_name


if __name__ == '__main__':

    file_name = 'concatenated_product_list2'

    # CSV.save_file(file_name)
    
    max_reviews_per_url = 20  # 각 URL당 최대 리뷰 개수
    CSV.save_file(file_name, max_reviews_per_url)
    print("크롤링 완료!")
