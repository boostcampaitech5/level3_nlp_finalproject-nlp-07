# from keybert import KeyBERT
# from transformers import AutoTokenizer, AutoModel
# from sklearn.cluster import KMeans
# import pandas as pd
# import numpy as np
# from keybert import KeyBERT
# from kiwipiepy import Kiwi
# from transformers import BertModel
# import pandas as pd
# import os
# import pandas as pd
# import pymysql
# import scipy.io
# import csv
# import pymysql
# import os
# import configparser
# from tqdm import tqdm



# if __name__ == "__main__":

#     # 환경 설정
#     env = os.getenv('MY_APP_ENV', 'local')  # 기본값은 'local'

#     def create_conn():
#         config = configparser.ConfigParser()
#         config.read(f'../config/config-{env}.ini')  # 환경에 맞는 설정 파일 읽기
#         mysql_config = config['mysql']
#         return pymysql.connect(host=mysql_config['host'], user=mysql_config['user'], password=mysql_config['password'],
#                             db=mysql_config['db'], charset=mysql_config['charset'])


#     # MySQL 연결 설정
#     conn = create_conn()

#     # MySQL query
#     query = "SELECT * FROM reviews_ver11_15_100"

#     # pandas를 사용하여 query 실행하고 DataFrame으로 결과 저장
#     reviews = pd.read_sql(query, conn)['context']

#     # 연결 종료
#     conn.close()




#     keywords = []
#     # 키워드 추출
#     kw_model = KeyBERT('distilbert-base-nli-mean-tokens')

#     for review in reviews:    
#         extracted_keywords = kw_model.extract_keywords(review)
#         keywords.extend(extracted_keywords)


#     print(keywords)


from keybert import KeyBERT
import pandas as pd
import os
import pymysql
import configparser

# 환경 설정
env = os.getenv('MY_APP_ENV', 'local')  # 기본값은 'local'

def create_conn():
    config = configparser.ConfigParser()
    config.read(f'../config/config-{env}.ini')  # 환경에 맞는 설정 파일 읽기
    mysql_config = config['mysql']
    return pymysql.connect(host=mysql_config['host'], user=mysql_config['user'], password=mysql_config['password'],
                        db=mysql_config['db'], charset=mysql_config['charset'])

# MySQL 연결 설정
conn = create_conn()

# MySQL query
query = "SELECT * FROM reviews_ver11_15_100"

# pandas를 사용하여 query 실행하고 DataFrame으로 결과 저장
reviews = pd.read_sql(query, conn)

# product_id별로 리뷰를 합침
grouped_reviews = reviews.groupby('prod_id')['context'].apply(' '.join).reset_index()

# 키워드 추출
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')

# 키워드 저장을 위한 빈 리스트 생성
keywords = []

for review in grouped_reviews['context']:    
    extracted_keywords = kw_model.extract_keywords(review, keyphrase_ngram_range=(1,1), stop_words='english', top_n=20)
    keywords.append(','.join([word for word, _ in extracted_keywords]))

# 키워드를 데이터프레임에 추가
grouped_reviews['keywords'] = keywords

# print(grouped_reviews)

# MySQL에 업데이트할 쿼리 생성
update_query = "UPDATE products_ver11_15_100 SET keywords = %s WHERE product_id = %s"

# 쿼리 실행 및 커밋
with conn.cursor() as cursor:
    for i, row in grouped_reviews.iterrows():
        cursor.execute(update_query, (row['keywords'], row['prod_id']))
    conn.commit()

# 연결 종료
conn.close()

print('Done!')