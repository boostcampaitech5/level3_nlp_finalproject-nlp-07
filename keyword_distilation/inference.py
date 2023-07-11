from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
import pandas as pd
import sqlite3
from keybert import KeyBERT
import pandas as pd
import os
import pymysql
import configparser
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, ndcg_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
import pandas as pd
import pymysql
import configparser
from keybert import KeyBERT

def compute_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def compute_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def compute_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

def compute_map_score(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

def compute_ndcg_score(true_scores, predicted_scores):
    return ndcg_score(true_scores, predicted_scores)

def compute_mrr_score(y_true, y_pred):
    scores = 0
    for t, p in zip(y_true, y_pred):
        if p in t:
            scores += 1 / (t.index(p) + 1)
    return scores / len(y_true)




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

# for _, evaluation in evaluations.iterrows():
#     user_query = evaluation['query']
#     search_name = evaluation['search_name']

#     # 쿼리 키워드 추출
#     query_keywords = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1,1), stop_words='english', top_n=20)
#     query_keywords = ','.join([word for word, _ in query_keywords])

#     # 데이터베이스에서 상품 정보 가져오기
#     query = f"SELECT product_id, keywords FROM products_ver12 WHERE search_name = '{search_name}'" 
#     products = pd.read_sql(query, conn)

#     # 쿼리와 각 상품 키워드간의 유사성 계산
#     vectorizer = CountVectorizer().fit_transform([query_keywords] + list(products['keywords']))
#     vectors = vectorizer.toarray()

#     csim = cosine_similarity(vectors)
#     query_similarities = csim[0, 1:]  # 첫 번째 행은 쿼리와 각 상품 사이의 유사성을 나타냅니다.

#     # 유사성에 따라 상품 정렬
#     product_similarities = list(zip(products['product_id'], query_similarities))
#     product_similarities.sort(key=itemgetter(1), reverse=True)

#     # 가장 유사한 상품 반환
#     recommended_product_id = product_similarities[0][0]
#     q = 'select * from products_ver12 where product_id = ' + str(recommended_product_id)
#     recommended_product = pd.read_sql(q, conn)
#     print("*"*100)
#     print(f"User query: {user_query}, Search name: {search_name}")
#     print(recommended_product['prod_name'])
#     print("*"*100)


import numpy as np

# for _, evaluation in evaluations.iterrows():
#     user_query = evaluation['query']
#     search_name = evaluation['search_name']
#     correct_product_ids = [int(id) for id in evaluation['product_id'].split(',')]
#     print(f"User query: {user_query}, Search name: {search_name}")

#     # 쿼리 키워드 추출
#     query_keywords = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1,1), stop_words='english', top_n=20)
#     query_keywords = ','.join([word for word, _ in query_keywords])

#     # 데이터베이스에서 상품 정보 가져오기
#     query = f"SELECT product_id, keywords FROM products_ver12 WHERE search_name = '{search_name}'" 
#     products = pd.read_sql(query, conn)

#     # 쿼리와 각 상품 키워드간의 유사성 계산
#     vectorizer = CountVectorizer().fit_transform([query_keywords] + list(products['keywords']))
#     vectors = vectorizer.toarray()

#     csim = cosine_similarity(vectors)
#     query_similarities = csim[0, 1:]  # 첫 번째 행은 쿼리와 각 상품 사이의 유사성을 나타냅니다.

#     # 유사성에 따라 상품 정렬
#     product_similarities = list(zip(products['product_id'], query_similarities))
#     product_similarities.sort(key=itemgetter(1), reverse=True)

#     # 상위 10개 상품의 id를 추출
#     recommended_product_ids = [id for id, _ in product_similarities[:10]]

#     # 평가 점수 계산
#     y_true = [1 if id in correct_product_ids else 0 for id in recommended_product_ids]
#     y_pred = [1] * len(y_true)
#     print(f"Precision: {compute_precision(y_true, y_pred)}")
#     print(f"Recall: {compute_recall(y_true, y_pred)}")
#     print(f"F1 Score: {compute_f1_score(y_true, y_pred)}")
#     print(f"MAP: {compute_map_score(y_true, y_pred)}")
#     print(f"NDCG: {compute_ndcg_score(np.asarray([y_true]), np.asarray([y_pred]))}")
#     print(f"MRR: {compute_mrr_score([y_true], [recommended_product_ids.index(correct_product_ids[0]) if correct_product_ids[0] in recommended_product_ids else len(recommended_product_ids)])}")


# def compute_mrr_score(rank_list):
#     scores = 0
#     for rank in rank_list:
#         scores += 1 / rank
#     return scores / len(rank_list)

# mrr_scores = []



# for _, evaluation in evaluations.iterrows():
#     user_query = evaluation['query']
#     search_name = evaluation['search_name']
#     correct_product_id = int(evaluation['product_id'])
#     print(f"User query: {user_query}, Search name: {search_name}")

#     # 쿼리 키워드 추출
#     query_keywords = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1,1), stop_words='english', top_n=20)
#     query_keywords = ','.join([word for word, _ in query_keywords])

#     # 데이터베이스에서 상품 정보 가져오기
#     query = f"SELECT product_id, keywords FROM products_ver12 WHERE search_name = '{search_name}'" 
#     products = pd.read_sql(query, conn)

#     # 쿼리와 각 상품 키워드간의 유사성 계산
#     vectorizer = CountVectorizer().fit_transform([query_keywords] + list(products['keywords']))
#     vectors = vectorizer.toarray()

#     csim = cosine_similarity(vectors)
#     query_similarities = csim[0, 1:]  # 첫 번째 행은 쿼리와 각 상품 사이의 유사성을 나타냅니다.

#     # 유사성에 따라 상품 정렬
#     product_similarities = list(zip(products['product_id'], query_similarities))
#     product_similarities.sort(key=itemgetter(1), reverse=True)

#     # 추천된 상품 ID 리스트
#     recommended_product_ids = [id for id, _ in product_similarities]

#     # MRR 계산
#     if correct_product_id in recommended_product_ids:
#         rank = recommended_product_ids.index(correct_product_id) + 1
#         mrr_scores.append(1 / rank)



#     # 가장 유사한 상품 반환
#     recommended_product_id = product_similarities[0][0]

#     # 평가 점수 계산
#     y_true = [1 if id == correct_product_id else 0 for id, _ in product_similarities[:10]]
#     y_pred = [1] * len(y_true)
#     print(f"Precision: {compute_precision(y_true, y_pred)}")
#     print(f"Recall: {compute_recall(y_true, y_pred)}")
#     print(f"F1 Score: {compute_f1_score(y_true, y_pred)}")
#     print(f"MAP: {compute_map_score(y_true, y_pred)}")


# # 전체 MRR 계산
# mean_mrr = compute_mrr_score(mrr_scores)
# print(f"Mean MRR: {mean_mrr}")


precision_scores = []
recall_scores = []
f1_scores = []
map_scores = []
mrr_scores = []

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

    # 추천된 상품 ID 리스트
    recommended_product_ids = [id for id, _ in product_similarities]

    # 평가 점수 계산
    y_true = [1 if id == correct_product_id else 0 for id, _ in product_similarities[:10]]
    y_pred = [1] * len(y_true)
    precision_scores.append(compute_precision(y_true, y_pred))
    recall_scores.append(compute_recall(y_true, y_pred))
    f1_scores.append(compute_f1_score(y_true, y_pred))
    map_scores.append(compute_map_score(y_true, y_pred))

    # MRR 계산
    if correct_product_id in recommended_product_ids:
        rank = recommended_product_ids.index(correct_product_id) + 1
        mrr_scores.append(1 / rank)


# 전체 평가 점수의 평균 계산
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)
mean_map = np.mean(map_scores)
mean_mrr = np.mean(mrr_scores)

print(f"Mean Precision: {mean_precision}")
print(f"Mean Recall: {mean_recall}")
print(f"Mean F1 Score: {mean_f1}")
print(f"Mean MAP: {mean_map}")
print(f"Mean MRR: {mean_mrr}")
