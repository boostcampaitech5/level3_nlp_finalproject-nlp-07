from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime
import pandas as pd
import requests
import json
import csv
import logging

def connect_to_mysql_and_insert_summary():
    # 1. MySQL로부터 데이터를 가져옴
    hook = MySqlHook(mysql_conn_id='my_first_test', charset='utf8mb4')
    connection = hook.get_conn()
    cursor = connection.cursor()
    cursor.execute("SELECT prod_id, SUBSTRING(GROUP_CONCAT(context SEPARATOR ' '), 1, 1500) AS context FROM reviews_ver31 GROUP BY prod_id")  # SQL query
    results = cursor.fetchall()
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
        if result[1] is None or result[1] == '' or result[1] == ' ':
            continue
        data = [result[1]]
        headers = {'Content-Type': 'application/json'}
        response = requests.post("http://49.50.166.224:30007/summary", data=json.dumps(data), headers=headers)
        summary = response.text
        summary = summary.encode('utf-8', 'ignore').decode('utf-8')

        # MySQL 커넥션을 열고
        hook = MySqlHook(mysql_conn_id='my_first_test', charset='utf8mb4')
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
