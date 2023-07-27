from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime
import pandas as pd
import requests
import json
import csv
import logging
from datetime import date

# 오늘 날짜를 YYYY-MM-DD 형식으로 변환
today = date.today().isoformat()


def fetch_and_store_reviews(prod_id, prod_name, search_name,  cursor, connection):
    url = f"http://49.50.166.224:30007/api/review/url?url={your_url}&prod_name={prod_name}&search_name={search_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # reviews 데이터를 MySQL에 저장하는 코드를 여기에 작성
            # call the function to write data to CSV
        # write_to_csv(data)\
        # data를 DataFrame으로 변환
        index = ['prod_name', 'user_name', 'rating', 'headline', 'review_content', 'answer', 'helped_cnt', 'top100_yn', 'search_name']
        # csv 직접 만들기

        # print(data)

        filename = "output.csv"

        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=index)
            writer.writeheader()
            for d in data:
                if d:  # Check if the list 'd' is not empty
                    for item in d:  # Iterate over each item in the list 'd'
                        if isinstance(item, dict):  # If 'item' is a dictionary
                            writer.writerow(item)
        
        for d in data:
            if d:  # Check if the list 'd' is not empty
                for item in d:  # Iterate over each item in the list 'd'
                    if isinstance(item, dict):  # If 'item' is a dictionary
                        # MySQL에 데이터 삽입
                        query = f"""
                        INSERT INTO reviews_ver31 (prod_id, user_name, rating, headline, review_content, answer, helped_cnt, top100_yn, search_name)
                        VALUES ({prod_id}, '{item['user_name']}', {item['rating']}, '{item['headline']}', '{item['review_content']}', '{item['answer']}', {item['helped_cnt']}, {item['top100_yn']}, '{item['search_name']}')
                        """
                        cursor.execute(query)

        connection.commit()
    else:
        print(f"Failed to get reviews for product id: {prod_id}")



def connect_to_mysql_and_insert_summary():
    # 1. MySQL로부터 데이터를 가져옴
    hook = MySqlHook(mysql_conn_id='my_first', charset='utf8mb4')
    connection = hook.get_conn()
    cursor = connection.cursor()

    
    # 오늘 날짜에 크롤링된 리뷰만 선택하는 쿼리
    query = f"SELECT prod_id, GROUP_CONCAT(context SEPARATOR ' ') AS context \
            FROM reviews_ver31 \
            WHERE DATE(crawl_date) = '{today}' \
            GROUP BY prod_id"
    
    cursor.execute(query)
    results = cursor.fetchall()
    logging.info(f"{today}")
    logging.info(f"Number of products: {len(results)}")

    print(results[0])

    with open("dags/go.csv", "w", encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([i[0] for i in cursor.description])
        csv_writer.writerows(results)
        f.flush()

        
    column_names = [desc[0] for desc in cursor.description]

    # df = pd.read_csv('dags/go.csv', names=column_names)
    # df.to_csv('dags/go.csv', index=False, encoding='utf-8')

    # 2. FastAPI를 통해 리뷰의 요약을 생성
    for result in results:
        # 리뷰 데이터가 부족한 제품 확인
        if result[1] is None or result[1] == '' or result[1] == ' ':
            prod_id = result[0]

            # 'review_count'를 조회하는 쿼리
            query = f"SELECT review_count FROM reviews_ver31 WHERE prod_id = {prod_id}"
            cursor.execute(query)
            review_count = cursor.fetchone()[0]

            # 리뷰 데이터가 10개 미만인 경우 API 호출로 리뷰 추가
            if review_count < 10:
                # 'prod_name'을 조회하는 쿼리
                query = f"SELECT url, prod_name, search_name FROM products_ver31 WHERE prod_id = {prod_id}"
                cursor.execute(query)
                url, prod_name, search_name = cursor.fetchone()
                
                # 리뷰를 추가하는 함수 호출
                fetch_and_store_reviews(url, prod_name, search_name, cursor, connection)
                logging.info(f"Added reviews for product id: {prod_id}")

                # 리뷰를 추가한 후 다시 쿼리를 실행하여 리뷰 데이터를 가져옴
                query = f"SELECT prod_id, GROUP_CONCAT(context SEPARATOR ' ') AS context \
                        FROM reviews_ver31 \
                        WHERE crawl_date = '{today}' AND prod_id = {prod_id} \
                        GROUP BY prod_id"
                cursor.execute(query)
                result = cursor.fetchone()
                data = [result[1]]


        


        data = [result[1]]
        headers = {'Content-Type': 'application/json'}
        response = requests.post("http://49.50.166.224:30007/summary", data=json.dumps(data), headers=headers)
        summary = response.text
        summary = summary.encode('utf-8', 'ignore').decode('utf-8')

        # MySQL 커넥션을 열고
        hook = MySqlHook(mysql_conn_id='my_first', charset='utf8mb4')
        connection = hook.get_conn()
        cursor = connection.cursor()

        # 요약을 MySQL에 추가
        update_query = f"UPDATE products_ver31 SET summary = '{summary}' WHERE product_id = {result[0]}"
        logging.info(update_query)
        cursor.execute(update_query)
        connection.commit()
        logging.info(f"Updated summary for product id: {result[0]}")

        # 커넥션을 닫음
        cursor.close()
        connection.close()



dag = DAG(
    'why_utf',
    start_date=datetime(2023, 7, 18),
    schedule_interval='0 4 * * *',  # This schedule means every 3 minutes
    tags=["why_utf"],
)

task = PythonOperator(
    task_id='connect_to_mysql_and_insert_summary',
    python_callable=connect_to_mysql_and_insert_summary,
    dag=dag,
)
