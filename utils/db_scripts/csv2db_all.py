import csv
import pymysql
import os
import configparser
from tqdm import tqdm

# 환경 설정
env = os.getenv('MY_APP_ENV', 'local')  # 기본값은 'local'

def create_conn():
    config = configparser.ConfigParser()
    config.read(f'../../config/config-{env}.ini')  # 환경에 맞는 설정 파일 읽기
    mysql_config = config['mysql']
    return pymysql.connect(host=mysql_config['host'], user=mysql_config['user'], password=mysql_config['password'],
                           db=mysql_config['db'], charset=mysql_config['charset'])


# MySQL 연결 설정
conn = create_conn()
curs = conn.cursor()


# 제품 데이터 삽입
with open('../crawler/data/products/떡볶이.csv', 'r', encoding='utf-8') as f_products:
    csvReader = csv.reader(f_products)
    next(csvReader)  # 헤더 건너뛰기
    for row in tqdm(csvReader):
        prod_name, price, url = row[0], row[1], row[3]
        sql = """
        INSERT INTO products 
        (prod_name, price, url)
        VALUES (%s, %s, %s)
        """
        curs.execute(sql, (prod_name, price, url))
        conn.commit()
        # print(row, "제품 삽입 완료")

# 리뷰 데이터 삽입
with open('../crawler/data/reviews/떡볶이리뷰.csv', 'r', encoding='utf-8') as f_reviews:
    csvReader = csv.reader(f_reviews)
    next(csvReader)  # 헤더 건너뛰기
    for row in tqdm(csvReader):
        prod_name, user_name, rating, title, context, answer, review_url = row[0], row[1], row[2], row[3], row[4], row[
            5], row[6]

        # 해당 제품의 product_id 가져오기
        curs.execute("SELECT product_id FROM products WHERE prod_name = %s", (prod_name,))
        result = curs.fetchone()
        if result is None:
            # print(f"No product found for name: {prod_name}")
            continue
        product_id = result[0]

        sql = """
        INSERT INTO reviews 
        (prod_id, prod_name, rating, title, context, answer, review_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        curs.execute(sql, (product_id, prod_name, rating, title, context, answer, review_url))
        conn.commit()
        # print(row, "리뷰 삽입 완료")

# 연결 종료
conn.close()
print("All Done!")
