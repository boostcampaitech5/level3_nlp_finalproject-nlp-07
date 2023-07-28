import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

def create_conn():
    db_uri = f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}"
    return create_engine(db_uri)

# MySQL 연결 설정
conn = create_conn()

# summary.csv 파일 읽기
summary_file = pd.read_csv('summary.csv')

# MySQL 연결 열기
with conn.begin() as transaction:
    # summary_file의 각 행을 순회
    for idx, row in summary_file.iterrows():
        # UPDATE 문 생성
        query = """
        UPDATE products_ver31 
        SET summary2 = :summary2
        WHERE product_id = :product_id
        """
        # UPDATE 문 실행
        transaction.execute(text(query), {"summary2": row['summary2'], "product_id": row['product_id']})

print("All updates were successfully applied!")
