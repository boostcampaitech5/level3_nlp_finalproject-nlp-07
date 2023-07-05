import scipy.io
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

conn.commit()

f = open('../crawler/data/products/떡볶이.csv', 'r', encoding='utf-8')

csvReader = csv.reader(f)

import codecs

i = 0
# f 순회
for row in tqdm(csvReader):
    i += 1
    if i == 1:
        continue

    prod_name = row[0]
    price = row[1]
    url = row[3]

    sql = """
    INSERT INTO products 
    (prod_name, price, url)
    VALUES (%s, %s, %s)
    """

    curs.execute(sql, (prod_name, price, url))
    product_id = curs.lastrowid  # Get the ID of the inserted row

    print(row, "삽입 완료")

conn.commit()
f.close()
conn.close()

print("All Done!")