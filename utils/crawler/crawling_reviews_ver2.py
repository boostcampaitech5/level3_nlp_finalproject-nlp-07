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
from preprocessing import preprocess

def get_headers(
    key: str,
    default_value: Optional[str] = None
    )-> Dict[str,Dict[str,str]]:
    """ Get Headers """
    JSON_FILE : str = './json/headers.json'

    with open(JSON_FILE,'r',encoding='UTF-8') as file:
        headers : Dict[str,Dict[str,str]] = json.loads(file.read())

    try :
        return headers[key]
    except:
        if default_value:
            return default_value
        raise EnvironmentError(f'Set the {key}')

class Coupang:

    MAX_REVIEWS_PER_URL = 20  # 각 URL당 최대 리뷰 개수

    @staticmethod
    def get_product_code(url: str)-> str:
        """ 입력받은 URL 주소의 PRODUCT CODE 추출하는 메소드 """
        prod_code : str = url.split('products/')[-1].split('?')[0]
        return prod_code

    def __init__(self)-> None:
        self.__headers : Dict[str,str] = get_headers(key='headers')

    def main(self, url_list: List[str]) -> List[List[Dict[str, Union[str, int]]]]:
        # 각 URL의 첫 페이지에서 리뷰를 가져옴
        result = []
        for URL in tqdm(url_list):
            # URL의 Product Code 추출
            prod_code: str = self.get_product_code(url=URL)
            review_counter = 0  # 각 URL별로 카운터 초기화

            review_counter = 0

            for page in range(1, 100):
                # URL 주소 재가공
                url_to_fetch = f'https://www.coupang.com/vp/product/reviews?productId={prod_code}&page={page}&size=10&sortBy=ORDER_SCORE_ASC&ratings=&q=&viRoleCode=3&ratingSummary=true'

                if review_counter >= self.MAX_REVIEWS_PER_URL:
                    break

                # __headers에 referer 키 추가
                self.__headers['referer'] = URL

                with rq.Session() as session:
                    page_data, review_counter = self.fetch(url=url_to_fetch, session=session, review_counter=review_counter)
                    result.append(page_data)

        return result

    def fetch(self,url:str,session, review_counter)-> List[Dict[str,Union[str,int]]]:
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

                # 평점
                rating = articles[idx].select_one('div.sdp-review__article__list__info__product-info__star-orange')
                if rating == None:
                    rating = 0
                else :
                    rating = int(rating.attrs['data-rating'])

                # 구매자 상품명
                prod_name = articles[idx].select_one('div.sdp-review__article__list__info__product-info__name')
                if prod_name == None or prod_name.text == '':
                    prod_name = '-'
                else:
                    prod_name = prod_name.text.strip()

                # 헤드라인(타이틀)
                headline = articles[idx].select_one('div.sdp-review__article__list__headline')
                if headline == None or headline.text == '':
                    headline = ''
                else:
                    headline = headline.text.strip()

                # 리뷰 내용
                review_content = articles[idx].select_one('div.sdp-review__article__list__review > div')
                if review_content == None :
                    review_content = ''
                else:
                    review_content = re.sub('[\n\t]','',review_content.text.strip())
                    review_content = preprocess(review_content)

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

                divs = soup.find_all('div', class_='sdp-review__article__list__survey__row')
                suryey_list = []
                # 각 div 요소에 대한 정보를 출력합니다
                for div in divs:
                    question = div.find('span', class_='sdp-review__article__list__survey__row__question').text
                    answer = div.find('span', class_='sdp-review__article__list__survey__row__answer').text
                    # print(f"Question: {question}, Answer: {answer}")
                    mix_text = f'<{question}>{answer}'
                    suryey_list.append(mix_text)
                survey = ''.join(suryey_list)
                # print(suryey)

                helped_cnt = articles[idx].select_one('.js_reviewArticleHelpfulContainer')
                if helped_cnt == None or helped_cnt.text == '':
                    helped_cnt = 0
                else:
                    help_cnt_str = helped_cnt.text.strip().split('명에게 도움 됨')[0]  # Split the string and get the first part
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
                dict_data['survey'] = survey

                review_counter += 1

                save_data.append(dict_data)

                # print(dict_data , '\n')

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
        
def load_urls(file_path: str) -> List[str]:
    # Load URLs from a CSV file
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        urls = [row['URL'] for row in reader]
    return urls


class CSV:
    @staticmethod
    def save_file()-> None:

        product_name = '치킨'

        file_path = f"./product_{product_name}_ver2.csv"
        url_list = load_urls(file_path)

        # 크롤링 결과
        results : List[List[Dict[str,Union[str,int]]]] = Coupang().main(url_list)

        # 파일에 쓸 데이터 준비
        csv_columns = ['prod_name', 'user_name', 'rating', 'headline', 'review_content', 'answer', 'helped_cnt', 'survey']
        # 서울 시간
        now = datetime.now(timezone('Asia/Seoul'))
        # 파일 이름
        # csv_file = f'../reviews/{product_name}_{now.strftime("%y%m%d_%H")}.csv'
        csv_file = f'./review_{product_name}_ver2.1.csv'

        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in results:
                for item in data:
                    writer.writerow(item)

        print(f'파일 저장완료!\n\n{csv_file}')


if __name__ == '__main__':

    # OpenPyXL.save_file()
    CSV.save_file()
