import pandas as pd
from selenium.webdriver.common.by import By

import datetime
from pytz import timezone
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys
from data.search_lists import search_products_list
from tqdm import tqdm



def crawling_products(search_list):
    '''
    리스트의 데이터를 크롤링하기
    :param search_list: 검색어 리스트
    :return: 크롤링한 데이터를 담은 리스트
    '''
    data = []

    name_list = []
    total_size = sum(len(v) for v in search_list.values())

    rank_list = ['number no-1 ', 'number no-2 ', 'number no-3 ', 'number no-4 ', 'number no-5 ', 'number no-6 ',
                 'number no-7 ', 'number no-8 ', 'number no-9 ', 'number no-10 ']

    for search_dict in tqdm(search_list, total=total_size):
        for search_name in search_list[search_dict]:
            # print(search_name)

            options = Options()
            options.add_argument("--window-size=300,100")  # 원하는 창 크기를 지정할 수 있습니다.

            driver = webdriver.Chrome(executable_path='chromedriver', options=options)

            url = f'https://www.coupang.com/np/search?component=&q={search_name}&channel=user'
            driver.get(url)

            lis = driver.find_elements(By.TAG_NAME, 'li')

            cnt = 0

            for li in lis:
                try:

                    if li.find_element(By.CLASS_NAME, 'number').get_attribute('class') not in rank_list:
                        continue

                    top_cnt = li.find_element(By.CLASS_NAME, 'number').get_attribute('class')
                    # # print(top_cnt)

                    # 제품명
                    name = li.find_element(By.CLASS_NAME, 'name').text
                    name = name.replace('\"', '')  # 쉼표 제거
                    name = search_name + ' ' + name

                    if name in name_list:
                        continue
                    name_list.append(name)

                    # 제품 고유번호
                    unique_product_id = li.find_element(By.CLASS_NAME, 'search-product-link').get_attribute(
                        'data-product-id')

                    # 가격
                    price = li.find_element(By.CLASS_NAME, 'price-value').text
                    price = price.replace(',', '')  # 쉼표 제거
                    price = price.replace('\"', '')  # 쌍따옴표 제거

                    # 리뷰수
                    review_cnt = li.find_element(By.CLASS_NAME, 'rating-total-count').text
                    review_cnt = review_cnt.replace('(', '').replace(')', '')  # 괄호 제거

                    # 리뷰수가 빈 문자열이 아니면 int로 변환
                    if review_cnt != '':
                        review_cnt = int(review_cnt)
                    else:
                        review_cnt = 0  # 또는 다른 기본값을 지정할 수 있습니다.

                    # 리뷰 평균 별점
                    rating = li.find_element(By.CLASS_NAME, 'rating').text
                    if rating != '':
                        rating = float(rating)
                    else:
                        rating = 0.0  # 또는 다른 기본값을 지정할 수 있습니다.

                    # 광고 여부
                    ad_elements = li.find_elements(By.CLASS_NAME, 'ad-badge-text')

                    # 제품 상세 페이지 URL
                    url = li.find_element(By.CLASS_NAME, 'search-product-link').get_attribute('href')

                    # TODO 재크롤링 필요한 정보들 (브랜드명, 브랜드점수)
                    # driver.get(url)

                    # 브랜드명
                    # brand_name = driver.find_element(By.CLASS_NAME, 'prod-sale-vendor-name').text

                    # 브랜드점수
                    # brand_rating = driver.find_element(By.CLASS_NAME, 'seller-rating with-company').text

                    if len(ad_elements) > 0:
                        ad_yn = 'Y'
                        # 광고 제품은 제외
                        continue
                    else:
                        ad_yn = 'N'

                    if review_cnt == '':
                        continue
                    if rating == '':
                        continue
                    if price == '':
                        continue
                    if name == '':
                        continue

                    # # # 리뷰 50개 이하는 제외
                    # if review_cnt <= 50:
                    #     continue

                    # if rating <= 3.0:
                    #     continue

                    data.append([search_name, unique_product_id, top_cnt, name, price, review_cnt, rating, ad_yn, url])
                    cnt += 1

                    if cnt >= 20:
                        break

                except Exception as e:
                    # print(e)
                    pass

            driver.quit()

    # Convert list of lists into dataframe
    df_output = pd.DataFrame(data,
                             columns=['search_name', 'unique_product_id', 'top_cnt', 'name', 'price', 'review_cnt',
                                      'rating', 'ad_yn', 'URL'])

    # time
    KST = datetime.datetime.now(timezone('Asia/Seoul')).strftime("%Y%m%d_%H%M%S")

    filename = f"product_list_{KST}"
    # Write dataframe to CSV
    df_output.to_csv(f'{filename}.csv', index=False)

    # Return filename as well
    return filename


if __name__ == '__main__':
    # 검색할 키워드 가져오기
    search_list = search_products_list()
    print(search_list)
    # 직접 넣도 싶다면 다음과 같은 형식으로 넣으면 된다.
    search_list = {'음식': ['감']}

    crawling_products(search_list)
