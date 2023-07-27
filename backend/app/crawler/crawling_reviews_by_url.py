import requests
from bs4 import BeautifulSoup
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
project_path = os.path.expanduser('~/level3_nlp_productserving-nlp-07/backend/app')
# Add the project path to the PYTHONPATH
sys.path.append(project_path)
from preprocessing.text_preprocessing import preprocess
from preprocessing.text_hanspell import spell_check
from pathlib import Path
import pandas as pd




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
    
def extract_url_reviews(url, prod_name, search_name):
    __headers = get_headers(key='headers')
    __headers['referer'] = url 

    result = []
    prod_code : str = url.split('products/')[-1].split('?')[0]
    review_counter = 0

    print(prod_code)

    for page in range(1, 100):
        # URL 주소 재가공
        url_to_fetch = f'https://www.coupang.com/vp/product/reviews?productId={prod_code}&page={page}&size=10&sortBy=ORDER_SCORE_ASC&ratings=&q=&viRoleCode=3&ratingSummary=true'

        if review_counter >= 10:
            break


        with rq.Session() as session:
            page_data, review_counter = fetch(url=url_to_fetch, session=session, review_counter=review_counter, prod_name=prod_name, search_name=search_name, __headers=__headers)
            result.append(page_data)

            # # review_content spell_check 처리 and write to CSV
            # for review in page_data:
            #     review['review_content'] = spell_check(review['review_content'])
            #     writer.writerow(review)



        # # review_content spell_check 처리
        # for page_data in result:
        #     for review in page_data:
        #         review['review_content'] = spell_check(review['review_content'])

    return result






def fetch(url:str,session, review_counter, prod_name:str, search_name:str, __headers:Dict[str,str]
          )-> List[Dict[str,Union[str,int]]]:
    save_data : List[Dict[str,Union[str,int]]] = list()

    with session.get(url=url,headers=__headers) as response :
        # print(response.status_code)
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
            if review_counter >= 10:
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
            # print(dict_data)

        return save_data, review_counter    

import ast
import csv

if __name__ == '__main__':

    data  = extract_url_reviews(
        'https://www.coupang.com/vp/products/166996432?itemId=478240933&vendorItemId=4200250100&pickType=COU_PICK', 
        '하', 
        '하')

    # call the function to write data to CSV
    # write_to_csv(data)\
    # data를 DataFrame으로 변환
    index = ['prod_name', 'user_name', 'rating', 'headline', 'review_content', 'answer', 'helped_cnt', 'top100_yn', 'search_name']
    # csv 직접 만들기

    # print(data)

    filename = "output.csv"

    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=index)
        writer.writeheader()
        for d in data:
            if d:  # Check if the list 'd' is not empty
                for item in d:  # Iterate over each item in the list 'd'
                    if isinstance(item, dict):  # If 'item' is a dictionary
                        writer.writerow(item)
