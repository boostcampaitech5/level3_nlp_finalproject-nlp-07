import pymysql
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import re
from starlette.middleware.cors import CORSMiddleware
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dotenv import load_dotenv

load_dotenv()  # .env 파일의 내용을 읽어서 환경변수로 설정


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


# MySQL 연결 설정
def create_conn():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        db=os.getenv("MYSQL_DB"),
        charset=os.getenv("MYSQL_CHARSET"),
    )


@app.get("/api/product/feedback_id/all")
async def get_feedback():
    conn = create_conn()
    curs = conn.cursor()
    curs.execute("SELECT * FROM feedback_data")
    results = curs.fetchall()
    conn.close()

    data = []
    for row in results:
        feedback_data = {
            "feedback_id": row[0],
            "query": row[1],
            "recommendations": row[2],
            "best": row[3],
        }
        data.append(feedback_data)

    return {"data": data}


@app.get("/api/product/summary/all")
async def get_all_summary():
    conn = create_conn()
    curs = conn.cursor()
    curs.execute(
        "SELECT product_id, unique_product_id, summary FROM products_ver31 WHERE summary IS NOT NULL"
    )
    results = curs.fetchall()
    conn.close()

    data = []
    for row in results:
        feedback_data = {
            "product_id": row[0],
            "unique_product_id": row[1],
            "summary": row[2],
        }
        data.append(feedback_data)

    return {"data": data}


@app.get("/api/product/feedback_id/{feedback_id}")
async def search_feedback(feedback_id: int):
    conn = create_conn()
    curs = conn.cursor()
    curs.execute("SELECT * FROM feedback_data WHERE feedback_id = %s", (feedback_id,))
    results = curs.fetchall()
    conn.close()

    if len(results) == 0:
        return {"error": f"No product found for name: {feedback_id}"}

    row = results[0]

    feedback_data = {
        "feedback_id": row[0],
        "query": row[1],
        "recommendations": row[2],
        "best": row[3],
    }

    return feedback_data


@app.post("/api/product/prod_id_list/")
async def search_product_list(prod_id_list: List[int]):
    # prod_id_list = ast.literal_eval(prod_id_list)
    prod_list = []
    for prod_id in prod_id_list:
        conn = create_conn()
        curs = conn.cursor()
        curs.execute(
            "SELECT product_id \
                        ,search_name \
                        ,summary \
                    FROM products_ver31 WHERE product_id = %s",
            (prod_id,),
        )
        results = curs.fetchall()
        conn.close()

        if len(results) == 0:
            return {"error": f"No product found for name: {prod_id}"}

        row = results[0]

        prod_data = {"product_id": row[0], "search_name": row[1], "summary": row[2]}
        prod_list.append(prod_data)

    return prod_list


@app.get("/api/product/prod_id_list/all/{prod_name}")
async def search_product_list(prod_name: str):
    # prod_id_list = ast.literal_eval(prod_id_list)

    conn = create_conn()
    curs = conn.cursor()
    curs.execute(
        "SELECT product_id \
                    ,search_name \
                    ,summary \
                FROM products_ver31 WHERE search_name = %s",
        (prod_name,),
    )
    results = curs.fetchall()
    conn.close()

    if len(results) == 0:
        return {"error": f"No product found for name: {prod_name}"}

    products = []
    for row in results:
        prod_data = {"product_id": row[0], "search_name": row[1], "summary": row[2]}
        products.append(prod_data)

    return {"product": products}


@app.get("/api/product/prod_id_list/all/{prod_name}")
async def search_product_list(prod_name: str):
    # prod_id_list = ast.literal_eval(prod_id_list)

    conn = create_conn()
    curs = conn.cursor()
    curs.execute(
        "SELECT product_id \
                    ,search_name \
                    ,summary \
                FROM products_ver31 WHERE search_name = %s",
        (prod_name,),
    )
    results = curs.fetchall()
    conn.close()

    if len(results) == 0:
        return {"error": f"No product found for name: {prod_name}"}

    products = []
    for row in results:
        prod_data = {"product_id": row[0], "search_name": row[1], "summary": row[2]}
        products.append(prod_data)

    return {"product": products}


@app.get("/api/review/all/")
async def search_summary_data():
    conn = create_conn()
    curs = conn.cursor()
    curs.execute("SELECT review_id, prod_id, context FROM reviews_ver31")
    results = curs.fetchall()
    conn.close()

    if len(results) == 0:
        return {"error": "Query Error"}

    reviews = []
    for row in results:
        prod_data = {
            "review_id": row[0],
            "prod_id": row[1],
            "context": row[2],
        }
        reviews.append(prod_data)

    return {"product": reviews}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
