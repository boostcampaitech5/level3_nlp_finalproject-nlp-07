import pymysql
import os
from fastapi import FastAPI
from pydantic import BaseModel
import configparser
import uvicorn
from typing import List, Union, Optional, Dict, Any
import aiomysql
import sys
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from crawler.crawling_products_bs4 import crawling_products
from crawler.crawling_reviews import CSV
from db_scripts.csv2db import run_pipeline
from pathlib import Path
import pandas as pd



app = FastAPI()


class Product(BaseModel):
    prod_name: str
    price: float
    url: str


class Review(BaseModel):
    prod_name: str
    prod_id: int
    rating: float
    title: str
    context: str
    answer: str
    review_url: str


# 환경 설정
env = os.getenv('MY_APP_ENV', 'local')  # 기본값은 'local'


# MySQL 연결 설정
def create_conn():
    config = configparser.ConfigParser()
    config.read(f'../../config/config-{env}.ini')  # 환경에 맞는 설정 파일 읽기
    mysql_config = config['mysql']
    print(mysql_config)
    return pymysql.connect(host=mysql_config['host'], user=mysql_config['user'], password=mysql_config['password'],
                           db=mysql_config['db'], charset=mysql_config['charset'])


@app.get("/product/prod_name/{prod_name}")
async def search_product(prod_name: str):
    '''
    제품명 검색 API (LIKE)
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    curs = conn.cursor()
    curs.execute("SELECT * FROM products WHERE prod_name LIKE %s", ('%' + prod_name + '%',))
    results = curs.fetchall()
    conn.close()

    if len(results) == 0:
        return {"error": f"No product found for name: {prod_name}"}

    products = []
    for row in results:
        products.append({"prod_name": row[1], "price": row[3], "url": row[4]})

    return {"products": products}


@app.get("/reviews/prod_name/{prod_name}")
def read_reviews(prod_name: str):
    '''
    정확히 일치하는 제품명 관련 리뷰 찾기 API (일치)
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews WHERE prod_name = %s", (prod_name,))
    result = cursor.fetchall()

    if len(result) == 0:
        conn.close()
        return {"error": f"No product found for name: {prod_name}"}

    reviews = []
    for row in result:
        reviews.append({
            "prod_id": row[1],
            "prod_name" : row[2],
            "rating": row[3],
            "title": row[4],
            "context": row[5],
            "answer": row[6],
            "review_url": row[7]
        })

    conn.close()
    return {"reviews": reviews}


@app.get("/reviews/search/prod_name/{prod_name}")
def read_reviews(prod_name: str):
    '''
    제품명으로 리뷰 검색 API (LIKE)
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews WHERE prod_name LIKE %s", ('%' + prod_name + '%',))
    result = cursor.fetchall()

    if len(result) == 0:
        conn.close()
        # 크롤링 로직
        # 직접 넣도 싶다면 다음과 같은 형식으로 넣으면 된다.
        print(prod_name)
        search_list = {'음식': [prod_name]}


        product_file_name = crawling_products(search_list)

        review_file_name = CSV.save_file(product_file_name)

        version = 'immediately_crawling'

        current_directory = Path(__file__).resolve().parent.parent.parent
        print(current_directory)

        product_csv_path = current_directory.joinpath("utils", "db_api", f"{product_file_name}.csv")
        review_csv_path = current_directory.joinpath("utils", "db_api", f"{review_file_name}.csv")

        product_csv_file = f"{product_csv_path}"
        review_csv_file = f"{review_csv_path}"

        run_pipeline(product_csv_file, review_csv_file, version)

        # print csv filenames
        print(os.path.basename(product_csv_file))
        print(os.path.basename(review_csv_file))

        review_df = pd.read_csv(review_csv_file)
        r = review_df['review_content']
        print(r)


        return {"crawling_yn":"Y", "reviews":r}
    
    reviews = []
    for row in result:
        reviews.append({
            "prod_id": row[1],
            "prod_name": row[2],
            "rating": row[3],
            "title": row[4],
            "context": row[5],
            "answer": row[6],
            "review_url": row[7]
        })

    conn.close()
    return {"crawling_yn":"N", "reviews": reviews}

@app.get("/reviews/all")
def read_reviews():
    '''
    리뷰 전부 가져오기
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews")
    result = cursor.fetchall()

    if len(result) == 0:
        conn.close()
        return

    reviews = []
    for row in result:
        reviews.append({
            "prod_id": row[1],
            "prod_name": row[2],
            "rating": row[3],
            "title": row[4],
            "context": row[5],
            "answer": row[6],
            "review_url": row[7]
        })

    conn.close()
    return {"reviews": reviews}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
