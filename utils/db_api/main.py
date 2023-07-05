import pymysql
import os
from fastapi import FastAPI
from pydantic import BaseModel
import configparser
import uvicorn
from typing import List, Union, Optional, Dict, Any
import aiomysql

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


@app.get("product/prod_name/{prod_name}")
async def search_product(prod_name: str):
    '''
    제품명 검색 API
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
        products.append({"prod_name": row[1], "price": row[2], "url": row[3]})

    return {"products": products}


@app.get("/reviews/prod_name/{prod_name}")
def read_reviews(prod_name: str):
    '''
    제품명으로 리뷰 검색 API
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
            "rating": row[3],
            "title": row[4],
            "context": row[5],
            "answer": row[6],
            "review_url": row[7]
        })

    conn.close()
    return {"reviews": reviews}

@app.get("/reviews/all")
def read_reviews():
    '''
    제품명으로 리뷰 검색 API
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
