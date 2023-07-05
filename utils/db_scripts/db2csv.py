import pandas as pd
import pymysql
import scipy.io
import csv
import pymysql
import os
import configparser
from tqdm import tqdm

# MySQL database 연결 정보 설정
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

# MySQL query
query = "SELECT * FROM reviews"

# pandas를 사용하여 query 실행하고 DataFrame으로 결과 저장
df = pd.read_sql(query, conn)

# 연결 종료
conn.close()

# DataFrame을 CSV 파일로 저장
df.to_csv('output.csv', index=False)