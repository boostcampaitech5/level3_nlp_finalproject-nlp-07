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
import os
import configparser
import pandas as pd
import pymysql
from keybert import KeyBERT
import matplotlib.pyplot as plt
import numpy as np

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

top_k = 3
hit_counts = []
mrr_scores = []

for _, evaluation in evaluations.iterrows():
    user_query = evaluation['query']
    search_name = evaluation['search_name']
    correct_product_id = int(evaluation['product_id'])

    # 쿼리 키워드 추출
    query_keywords = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1,1), stop_words='english')
    query_keywords = [word for word, _ in query_keywords]
    
    # 데이터베이스에서 상품 정보 가져오기
    query = f"SELECT product_id, keywords FROM products_ver12 WHERE search_name = '{search_name}'" 
    products = pd.read_sql(query, conn)

    # 쿼리와 각 상품 키워드간의 유사성 계산
    vectorizer = CountVectorizer().fit_transform([user_query] + list(products['keywords']))
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)
    query_similarities = csim[0, 1:]  # 첫 번째 행은 쿼리와 각 상품 사이의 유사성을 나타냅니다.

    # 유사성에 따라 상품 정렬
    product_similarities = list(zip(products['product_id'], query_similarities))
    product_similarities.sort(key=itemgetter(1), reverse=True)

    # 추천된 상품 ID 리스트
    recommended_product_ids = [id for id, _ in product_similarities]

    # Top-k Hit 확인
    if correct_product_id in recommended_product_ids[:top_k]:
        hit_counts.append(1)
        
        # MRR 계산
        rank = recommended_product_ids.index(correct_product_id) + 1
        mrr_scores.append(1 / rank)
    else:
        hit_counts.append(0)

# 평가 점수 계산
top_k_hit_rate = sum(hit_counts) / len(hit_counts)
mean_mrr = sum(mrr_scores) / len(mrr_scores)

print(f"Top-{top_k} Hit Rate: {top_k_hit_rate}")
print(f"Mean MRR: {mean_mrr}")

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.bar(['Top-k Hit Rate', 'Mean MRR'], [top_k_hit_rate, mean_mrr])
plt.ylabel('Score')
plt.title('Evaluation Results')
plt.savefig('evaluation_results.png')
