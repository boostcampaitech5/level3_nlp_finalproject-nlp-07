import pymysql
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Union, Optional, Dict, Any
import re
from starlette.middleware.cors import CORSMiddleware
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from crawler.crawling_products_bs4 import crawling_products
from crawler.crawling_reviews import CSV
from db_scripts.csv2db import run_pipeline
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from dotenv import load_dotenv
load_dotenv()  # .env 파일의 내용을 읽어서 환경변수로 설정
import requests
from typing import Optional
from crawler.crawling_price import extract_price
from crawler.crawling_reviews_by_url import extract_url_reviews
from fastapi import FastAPI, Body
import time



app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
env = os.getenv("MY_APP_ENV", "local")  # 기본값은 'local'


def dict_to_list(reviews):
    reviews = [review["context"] for review in reviews]
    reviews = [re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", text) for text in reviews]
    reviews = [re.sub(r"[^가-힣a-zA-Z0-9\n\s]", "", text).strip() for text in reviews]

    filtered_reviews = []
    for i in range(10):
        filtered_reviews.append(" ".join(reviews[i * 20 : i * 20 + 20]))

    return filtered_reviews


# MySQL 연결 설정
def create_conn():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        db=os.getenv("MYSQL_DB"),
        charset=os.getenv("MYSQL_CHARSET"),
    )


@app.get("/api/product/prod_name/{prod_name}")
async def search_product(prod_name: str):
    '''
    제품명 검색 API (LIKE)
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    curs = conn.cursor()
    curs.execute("SELECT product_id \
                    ,unique_product_id \
                    ,top_cnt \
                    ,search_name \
                    ,prod_name \
                    ,price \
                    ,url \
                    ,avg_rating \
                    ,review_cnt \
                    ,ad_yn \
                    ,product_img_url \
                 FROM products_ver31 WHERE prod_name LIKE %s", ('%' + prod_name + '%',))
    results = curs.fetchall()
    conn.close()

    if len(results) == 0:
        return {"error": f"No product found for name: {prod_name}"}

    products = []
    for row in results:
        # search_name,unique_product_id,top_cnt,name,price,review_cnt,rating,ad_yn,URL,product_img_url
        products.append({
            "product_id": row[0],
            "unique_product_id": row[1],
            "top_cnt": row[2],
            "search_name": row[3],
            "prod_name": row[4],
            "price": row[5],
            "url": row[6],
            "avg_rating": row[7],
            "review_cnt": row[8],
            "ad_yn": row[9],
            "product_img_url": row[10]
        })


    return {"products": products}


@app.get("/api/reviews/prod_name/{prod_name}")
def read_reviews(prod_name: str):
    '''
    정확히 일치하는 제품명 관련 리뷰 찾기 API (일치)
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews_ver31 WHERE search_name = %s", (prod_name,))
    result = cursor.fetchall()

    if len(result) == 0:
        print("No review found for name: ", prod_name)
        # 크롤링 로직
        # 직접 넣도 싶다면 다음과 같은 형식으로 넣으면 된다.
        print(prod_name)
        search_list = {'음식': [prod_name]}


        product_file_name = crawling_products(search_list)

        review_file_name = CSV.save_file(product_file_name, 3)

        version = 'ver31'

        # current_directory = Path(__file__).resolve().parent.parent.parent
        current_directory = Path(__file__).resolve().parent.parent
        # print(current_directory)

        # product_csv_path = current_directory.joinpath("backend", "app", f"{product_file_name}.csv")
        # review_csv_path = current_directory.joinpath("backend", "app", f"{review_file_name}.csv")
        product_csv_path = current_directory.joinpath("app", f"{product_file_name}.csv")
        review_csv_path = current_directory.joinpath("app", f"{review_file_name}.csv")

        product_csv_file = f"{product_csv_path}"
        review_csv_file = f"{review_csv_path}"

        run_pipeline(product_csv_file, review_csv_file, version)

        # print csv filenames
        # print(os.path.basename(product_csv_file))
        # print(os.path.basename(review_csv_file))

        cursor.execute("SELECT * FROM reviews_ver31 WHERE search_name = %s", (prod_name,))
        result = cursor.fetchall()
        # print("result", result)


        # product_df = pd.read_csv(product_csv_file)
        # product_df = product_df.replace([np.inf, -np.inf], np.nan)  # replace all inf by NaN
        # product_df = product_df.dropna()  # drop all rows with NaN

        # review_df = pd.read_csv(review_csv_file, dtype={"headline": str})
        # review_df = review_df.replace([np.inf, -np.inf], np.nan)  # replace all inf by NaN
        # review_df = review_df.dropna()  # drop all rows with NaN


        # products = []
        # for index, row in product_df.iterrows():
        #     products.append({
        #             "search_name": row[0], # 검색어
        #             "unique_product_id": row[1], # 쿠팡 상품 고유 ID
        #             "top_cnt": row[2], # 쿠팡 랭킹
        #             "prod_name": row[3],
        #             "price": row[4], # 가격
        #             "review_cnt": row[5], # 리뷰 수
        #             "rating": row[6], # 평균 평점
        #             "ad_yn": row[7], # 광고 여부
        #             "URL": row[8], # 상품 URL
        #             "product_img_url" : row[9] # 상품 이미지 URL
        #         })



        # reviews = []
        # for index, row in review_df.iterrows():
        #     reviews.append({
        #         "prod_name" : row[0],
        #         "rating": row[2],
        #         "title": row[3],
        #         "context": row[4],
        #         "answer": row[5]
        #     })  

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
        return {"source":"crawl", "reviews":reviews}

    products = []

    cursor.execute("SELECT * FROM products_ver31 WHERE prod_name = %s", (prod_name,))
    
    result = cursor.fetchall()

    for row in result:
        products.append({
            "product_id": row[0],
            "prod_name" : row[4],
            "price": row[6],
            "url":row[7],
            "summary":row[15],
            "product_img_url":row[17],
        })

    conn.close()
    return {"source":"db", "products":products} 


@app.get("/api/reviews/search/prod_name/{prod_name}")
def read_reviews(prod_name: str):
    '''
    제품명으로 리뷰 검색 API (LIKE)
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews_ver31 WHERE search_name = %s", (prod_name))
    result = cursor.fetchall()


    if len(result) == 0:
        print("No review found for name: ", prod_name)
        # 크롤링 로직
        # 직접 넣도 싶다면 다음과 같은 형식으로 넣으면 된다.
        print(prod_name)
        search_list = {'음식': [prod_name]}


        product_file_name = crawling_products(search_list)

        review_file_name = CSV.save_file(product_file_name, 3)

        version = 'ver31'

        # current_directory = Path(__file__).resolve().parent.parent.parent
        current_directory = Path(__file__).resolve().parent.parent
        # print(current_directory)

        # product_csv_path = current_directory.joinpath("backend", "app", f"{product_file_name}.csv")
        # review_csv_path = current_directory.joinpath("backend", "app", f"{review_file_name}.csv")
        product_csv_path = current_directory.joinpath("app", f"{product_file_name}.csv")
        review_csv_path = current_directory.joinpath("app", f"{review_file_name}.csv")

        product_csv_file = f"{product_csv_path}"
        review_csv_file = f"{review_csv_path}"

        run_pipeline(product_csv_file, review_csv_file, version)

        # print csv filenames
        # print(os.path.basename(product_csv_file))
        # print(os.path.basename(review_csv_file))
        conn.close()

        conn = create_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reviews_ver31 WHERE search_name = %s", (prod_name))
        result = cursor.fetchall()
        # print("result", result)


        # product_df = pd.read_csv(product_csv_file)
        # product_df = product_df.replace([np.inf, -np.inf], np.nan)  # replace all inf by NaN
        # product_df = product_df.dropna()  # drop all rows with NaN

        # review_df = pd.read_csv(review_csv_file, dtype={"headline": str})
        # review_df = review_df.replace([np.inf, -np.inf], np.nan)  # replace all inf by NaN
        # review_df = review_df.dropna()  # drop all rows with NaN


        # products = []
        # for index, row in product_df.iterrows():
        #     products.append({
        #             "search_name": row[0], # 검색어
        #             "unique_product_id": row[1], # 쿠팡 상품 고유 ID
        #             "top_cnt": row[2], # 쿠팡 랭킹
        #             "prod_name": row[3],
        #             "price": row[4], # 가격
        #             "review_cnt": row[5], # 리뷰 수
        #             "rating": row[6], # 평균 평점
        #             "ad_yn": row[7], # 광고 여부
        #             "URL": row[8], # 상품 URL
        #             "product_img_url" : row[9] # 상품 이미지 URL
        #         })



        # reviews = []
        # for index, row in review_df.iterrows():
        #     reviews.append({
        #         "prod_name" : row[0],
        #         "rating": row[2],
        #         "title": row[3],
        #         "context": row[4],
        #         "answer": row[5]
        #     })  

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
        
        return {"source":"crawl", "reviews":reviews}
    


    products = []

    cursor.execute("SELECT * FROM products_ver31 WHERE search_name = %s", (prod_name))
    result = cursor.fetchall()

    for row in result:
        products.append({
            "product_id": row[0],
            "prod_name" : row[4],
            "price": row[6],
            "url":row[7],
            "summary":row[15],
            "product_img_url":row[17],
        })

    conn.close()
    return {"source":"db", "products":products} 


@app.get("/api/reviews/all")
def read_reviews():
    '''
    리뷰 전부 가져오기
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews_ver31")
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



class ProductIds(BaseModel):
    product_id: List[int]

@app.post("/api/products/url")
async def get_products(product_ids: ProductIds):
    conn = create_conn()
    cursor = conn.cursor()
    
    # SQL injection을 피하기 위해 매개변수를 안전하게 전달
    query = "SELECT product_id, url, product_img_url FROM products_ver31 WHERE product_id in ({})".format(', '.join(['%s'] * len(product_ids.product_id)))
    cursor.execute(query, tuple(product_ids.product_id))
    rows = cursor.fetchall()

    data = {}
    for idx, row in enumerate(rows):
        data[f'prod_id{idx+1}'] = {  # row[0] is the product_id
            "url": row[1],  # row[1] is the url
            "product_img_url": row[2]  # row[2] is the product_img_url
        }
    
    # 데이터베이스 연결 종료
    cursor.close()
    conn.close()
    
    # code와 data를 포함한 결과 반환
    result = data
    try:
        return result
    except:
        raise HTTPException(status_code=801, detail="products url Error")




class FeedbackIn(BaseModel):
    query: str
    recommendations: str
    best: Optional[str] = None
    review: Optional[str] = None


class FeedbackOut(FeedbackIn):
    feedback_id: int

class UpdateData(BaseModel):
    review: str



@app.post("/api/feedback/")
async def create_feedback(feedback: FeedbackIn):
    conn = None
    cursor = None
    try:
        conn = create_conn()
        cursor = conn.cursor()
        cursor.execute("insert into feedback_data(query, recommendations, best, review) values(%s, %s, %s, %s)", 
                       (feedback.query, feedback.recommendations, feedback.best, feedback.review))

        conn.commit()
        last_row_id = cursor.lastrowid
    except Exception as e:
        if conn is not None:
            conn.rollback()   # rollback to previous state
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    try:
        return FeedbackOut(**feedback.dict(), feedback_id=last_row_id)
    except:
        raise HTTPException(status_code=701, detail="feedback insert Error")



@app.put("/api/feedback/feedback_id/{feedback_id}")
async def update_feedback(feedback_id: int, update_data: UpdateData):
    conn = None
    cursor = None
    try:
        conn = create_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE feedback_data SET review = %s WHERE feedback_id = %s",
                       (update_data.review, feedback_id))

        conn.commit()

    except Exception as e:
        if conn is not None:
            conn.rollback()   # rollback to previous state
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    try:
        return {"feedback_id": feedback_id}
    except:
        raise HTTPException(status_code=801, detail="feedback update Error")


# 크롤링 하는 api
@app.get("/api/crawl/{prod_name}")
def read_reviews(prod_name: str):
    '''
    제품명으로 리뷰 검색 API (LIKE)
    :param prod_name:
    :return:
    '''
    conn = create_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reviews_ver100 WHERE search_name = %s", (prod_name))
    result = cursor.fetchall()


    if len(result) == 0:
        print("No review found for name: ", prod_name)
        # 크롤링 로직
        # 직접 넣도 싶다면 다음과 같은 형식으로 넣으면 된다.
        print(prod_name)
        search_list = {'음식': [prod_name]}


        product_file_name = crawling_products(search_list)

        review_file_name = CSV.save_file(product_file_name, 3)

        version = 'ver100'

        # current_directory = Path(__file__).resolve().parent.parent.parent
        current_directory = Path(__file__).resolve().parent.parent
        # print(current_directory)

        product_csv_path = current_directory.joinpath("app", f"{product_file_name}.csv")
        review_csv_path = current_directory.joinpath("app", f"{review_file_name}.csv")

        product_csv_file = f"{product_csv_path}"
        review_csv_file = f"{review_csv_path}"

        run_pipeline(product_csv_file, review_csv_file, version)

        # print csv filenames
        # print(os.path.basename(product_csv_file))
        # print(os.path.basename(review_csv_file))
        conn.close()

        conn = create_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reviews_ver100 WHERE search_name = %s", (prod_name))
        result = cursor.fetchall()
        # print("result", result)

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
        
        return {"source":"crawl", "reviews":reviews}
    


    products = []

    cursor.execute("SELECT * FROM products_ver100 WHERE search_name = %s", (prod_name))
    result = cursor.fetchall()

    for row in result:
        products.append({
            "product_id": row[0],
            "prod_name" : row[4],
            "price": row[6],
            "url":row[7],
            "summary":row[15],
            "product_img_url":row[17],
        })

    conn.close()
    return {"source":"db", "products":products} 



@app.get("/api/price/")
def read_reviews(url: Optional[str] = None):
    '''
    url로 제품 가격 크롤링
    :param url:
    :return:
    
    '''

    if url is None:
        return {"detail": "No url provided"}

    price = extract_price(url)
    return {"price": price}

class Item(BaseModel):
    prod_names: List[str]



@app.post("/api/crawl/")
def read_reviews(item: Item):
     # 각 상품에 대해 시간을 측정하고 저장할 리스트를 준비합니다.
    elapsed_times = []

    prod_names = item.prod_names
    
    # 각 상품에 대해 크롤링을 수행합니다.
    for prod_name in prod_names:
        # 시간 측정 시작
        start_time = time.time()

       

        conn = create_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reviews_ver100 WHERE search_name = %s", (prod_name))
        result = cursor.fetchall()


        if len(result) == 0:
            print("No review found for name: ", prod_name)
            # 크롤링 로직
            # 직접 넣도 싶다면 다음과 같은 형식으로 넣으면 된다.
            print(prod_name)
            search_list = {'음식': [prod_name]}


            product_file_name = crawling_products(search_list)

            review_file_name = CSV.save_file(product_file_name, 3)

            version = 'ver100'

            # current_directory = Path(__file__).resolve().parent.parent.parent
            current_directory = Path(__file__).resolve().parent.parent
            # print(current_directory)

            product_csv_path = current_directory.joinpath("app", f"{product_file_name}.csv")
            review_csv_path = current_directory.joinpath("app", f"{review_file_name}.csv")

            product_csv_file = f"{product_csv_path}"
            review_csv_file = f"{review_csv_path}"

            run_pipeline(product_csv_file, review_csv_file, version)

            # print csv filenames
            # print(os.path.basename(product_csv_file))
            # print(os.path.basename(review_csv_file))
            conn.close()

            conn = create_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reviews_ver100 WHERE search_name = %s", (prod_name))
            result = cursor.fetchall()
            # print("result", result)

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

            # 시간 측정 종료 및 걸린 시간 계산
            end_time = time.time()
            elapsed_time = end_time - start_time

            elapsed_times.append({"product_name": prod_name, "elapsed_time": elapsed_time})

            
            return {"source":"crawl", "reviews":reviews}
        


        products = []

        cursor.execute("SELECT * FROM products_ver100 WHERE search_name = %s", (prod_name))
        result = cursor.fetchall()

        for row in result:
            products.append({
                "product_id": row[0],
                "prod_name" : row[4],
                "price": row[6],
                "url":row[7],
                "summary":row[15],
                "product_img_url":row[17],
            })
         # 시간 측정 종료 및 걸린 시간 계산
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append({"product_name": prod_name, "elapsed_time": elapsed_time})

    

        conn.close()

    avg_time = sum([time["elapsed_time"] for time in elapsed_times]) / len(elapsed_times)
    return {"elapsed_times": elapsed_times, "avg_time":avg_time}

   


@app.get("/api/review/url")
def read_reviews(url: Optional[str] = None, prod_name: Optional[str] = None, search_name: Optional[str] = None):
    '''
    url로 리뷰 가격 크롤링
    :param url:
    :return:
    
    '''

    if url is None:
        return {"detail": "No url provided"}

    data = extract_url_reviews(url, prod_name, search_name)
    return {"data": data}



# uvicorn main:app --port 30008 --host 0.0.0.0
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)  