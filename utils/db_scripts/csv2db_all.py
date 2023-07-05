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
with open('../crawler/product_떡볶이_ver2.csv', 'r', encoding='utf-8') as f_products:
    csvReader = csv.reader(f_products)
    next(csvReader)  # 헤더 건너뛰기
    for row in tqdm(csvReader):
        prod_name, price, review_cnt, avg_rating, ad_yn, url = row[0], row[1], row[2], row[3], row[4], row[5]

        sql = """
        INSERT INTO products_ver2
        (prod_name, price, review_cnt, avg_rating, ad_yn, url)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        curs.execute(sql, (prod_name, price, review_cnt, avg_rating, ad_yn, url))
        conn.commit()

# 리뷰 데이터 삽입
with open('../crawler/review_떡볶이_ver2.csv', 'r', encoding='utf-8') as f_reviews:
    csvReader = csv.reader(f_reviews)
    next(csvReader)  # 헤더 건너뛰기
    for row in tqdm(csvReader):

        prod_name, user_name, rating, title, context, answer, helped_cnt = row[0], row[1], row[2], row[3], row[4], row[5], row[6]

        # 해당 제품의 product_id 가져오기
        curs.execute("SELECT product_id FROM products_ver2 WHERE prod_name = %s", (prod_name,))
        result = curs.fetchone()
        if result is None:
            # print(f"No product found for name: {prod_name}")
            continue
        product_id = result[0]

        sql = """
        INSERT INTO reviews_ver2
        (prod_id, prod_name, rating, title, context, answer, helped_cnt)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        curs.execute(sql, (product_id, prod_name, rating, title, context, answer, helped_cnt))
        conn.commit()
        # print(row, "리뷰 삽입 완료")

# 연결 종료
conn.close()
print("All Done!")
