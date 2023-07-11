import pandas as pd
import os
import pymysql
import configparser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
from keybert import KeyBERT
import numpy as np
import matplotlib.pyplot as plt

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

query = "SELECT query, search_name, product_id FROM evaluete_retrieve"
evaluations = pd.read_sql(query, conn)

# 키워드 추출 모델
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')

precision_scores = []
top3_hits = 0

for _, evaluation in evaluations.iterrows():
    
    user_query = evaluation['query']
    search_name = evaluation['search_name']
    correct_product_id = int(evaluation['product_id'])

    # 쿼리 키워드 추출
    query_keywords = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1,1), stop_words='english', top_n=20)
    query_keywords = ','.join([word for word, _ in query_keywords])

    # 데이터베이스에서 상품 정보 가져오기
    query = f"SELECT product_id, keywords FROM products_ver12 WHERE search_name = '{search_name}'" 
    products = pd.read_sql(query, conn)

    # 쿼리와 각 상품 키워드간의 유사성 계산
    vectorizer = CountVectorizer().fit_transform([query_keywords] + list(products['keywords']))
    vectors = vectorizer.toarray()

    csim = cosine_similarity(vectors)
    query_similarities = csim[0, 1:]  # 첫 번째 행은 쿼리와 각 상품 사이의 유사성을 나타냅니다.

    # 유사성에 따라 상품 정렬
    product_similarities = list(zip(products['product_id'], query_similarities))
    product_similarities.sort(key=itemgetter(1), reverse=True)

    # 평가 점수 계산
    y_true = [1 if id == correct_product_id else 0 for id, _ in product_similarities[:3]]  # Top 5
    precision_scores.append(sum(y_true) / len(y_true))  # Precision@5

    # Top 3 hit count
    top3_products = [id for id, _ in product_similarities[:3]]  # Top 3
    if correct_product_id in top3_products:
        # prod name 을 가져와야함


        top3_hits += 1

# 전체 평가 점수의 평균 계산
mean_precision = np.mean(precision_scores)
top3_hit_rate = top3_hits / len(evaluations)

print(f"Mean Precision@3: {mean_precision}")
print(f"Top 3 Hit Rate: {top3_hit_rate}")

# Precision@5 scores chart
plt.figure(figsize=(10, 5))
plt.plot(precision_scores, marker='o')
plt.title('Precision@3 scores for each query')
plt.xlabel('Query Index')
plt.ylabel('Precision@3')
plt.grid(True)
plt.savefig('precision_at_3.png')
