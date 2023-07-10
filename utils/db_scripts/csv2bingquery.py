import pandas as pd
import os
from google.cloud import bigquery
import configparser

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./salmons.json"

# BigQuery 클라이언트 생성
client = bigquery.Client()
# 환경 설정
env = os.getenv('MY_APP_ENV', 'local')  # 기본값은 'local'

config = configparser.ConfigParser()
config.read(f'../../config/config-{env}.ini')  # 환경에 맞는 설정 파일 읽기
bigquery_config = config['bigquery']
project_id = bigquery_config['project_id']



# 데이터셋 ID 생성 (여기서 "my_project"는 본인의 프로젝트 ID로 대체)
# dataset_id = "salmons-391809.salmon".format(client.project)

# # 데이터셋 생성
# dataset = bigquery.Dataset(dataset_id)
# dataset = client.create_dataset(dataset)  # API 요청
#
# print("Dataset salmon created.".format(dataset_id))
#


# 제품 데이터 삽입
df_products = pd.read_csv('../crawler/product_떡볶이_ver5.csv', encoding='utf-8')
table_id = f"{project_id}.salmon.products_ver5" # 변경 필요
job = client.load_table_from_dataframe(df_products, table_id)
job.result()  # Wait for the job to complete.

# 리뷰 데이터 삽입
df_reviews = pd.read_csv('../crawler/review_떡볶이_ver5.csv', encoding='utf-8')
table_id = f"{project_id}.salmon.reviews_ver5" # 변경 필요
job = client.load_table_from_dataframe(df_reviews, table_id)
job.result()  # Wait for the job to complete.

print("All Done!")
