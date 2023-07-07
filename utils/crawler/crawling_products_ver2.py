import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import datetime
from pytz import timezone


def product_one_crawler(search_name):
    driver = webdriver.Chrome(executable_path='chromedriver')

    data = []
    name_list = [] # 중복 체크용

    url = f'https://www.coupang.com/np/search?component=&q={search_name}&channel=user'
    driver.get(url)

    lis = driver.find_elements(By.TAG_NAME, 'li')

    cnt = 0

    for li in lis:
        try:

            # 제품명
            name = li.find_element(By.CLASS_NAME, 'name').text
            name = name.replace('\"', '')  # 쉼표 제거

            # 중복 제거
            if name in name_list:
                continue
            name_list.append(name)

            # 제품 고유번호
            # unique_product_id = li.find_element(By.CLASS_NAME, 'search-product').get_attribute('data-product-id')

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

            # # 리뷰 50개 이하는 제외
            if review_cnt <= 50:
                continue

            if rating <= 3.0:
                continue

            data.append([name, price, review_cnt, rating, ad_yn, url])
            cnt += 1

            if cnt == 10:
                break

        except Exception as e:
            pass

    driver.quit()

    # Convert list of lists into dataframe
    df_output = pd.DataFrame(data, columns=['name', 'price', 'review_cnt', 'rating', 'ad_yn', 'URL'])

    # time
    KST = datetime.datetime.now(timezone('Asia/Seoul')).strftime("%Y%m%d_%H%M%S")
    # Write dataframe to CSV
    df_output.to_csv(f"product_{KST}_{search_name}.csv", index=False)


if __name__ == '__main__':
    search_name = '떡볶이'
    product_one_crawler(search_name)
