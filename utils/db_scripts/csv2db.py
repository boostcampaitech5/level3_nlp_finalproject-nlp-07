import csv
import pymysql
import os
import configparser
from tqdm import tqdm
from pathlib import Path

# 환경 설정
env = os.getenv('MY_APP_ENV', 'local')  # 기본값은 'local'


def check_products_table(cursor, table_name):
    cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS products_{table_name} (
                `product_id`           int NOT NULL AUTO_INCREMENT COMMENT '제품 ID',
                `unique_product_id`    varchar(255) COMMENT '제품 고유 ID',
                `top_cnt`              varchar(10) COMMENT '쿠팡 TOP10 랭킹',
                `search_name`          varchar(255) COMMENT '검색어',
                `prod_name`            varchar(255) COMMENT '제품 이름',
                `description`          text COMMENT '제품에 대한 설명',
                `price`                varchar(500) COMMENT '제품의 가격',
                `url`                  varchar(255) COMMENT '상품 URL',
                `create_date`          varchar(255) COMMENT '등록일',
                `avg_rating`           int COMMENT '평균 평점',
                `brand_name`           varchar(255) COMMENT '판매사',
                `review_cnt`           int COMMENT '리뷰 수',
                `ad_yn`                varchar(1) COMMENT '광고 제품 여부',
                `positive_reviews_cnt` int COMMENT '긍정 리뷰 수',
                `negative_reviews_cnt` int COMMENT '부정 리뷰 수',
                `summary`              text COMMENT '제품 요약(모델링)',
                `keywords`             text COMMENT '키워드 추출(모델링)',
                PRIMARY KEY (`product_id`)
            ) COMMENT = '제품 정보 테이블'
        """)


def check_reviews_table(cursor, table_name):
    cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS reviews_{table_name} (
                `review_id`       int NOT NULL AUTO_INCREMENT COMMENT '리뷰 ID',
                `prod_id`         int COMMENT '제품 ID',
                `prod_name`       varchar(255) COMMENT '제품 이름',
                `rating`          varchar(10) COMMENT '리뷰 평점',
                `title`           varchar(255) COMMENT '리뷰 제목',
                `context`         text COMMENT '리뷰 내용',
                `answer`          varchar(255) COMMENT '리뷰 총평',
                `review_url`      varchar(255) COMMENT '리뷰 URL',
                `helped_cnt`      int COMMENT '도움이 된 수',
                `create_date`     varchar(255) COMMENT '작성일',
                `top100_yn`       varchar(1) COMMENT '리뷰작성자 TOP 100 여부',
                `sentiment`       varchar(1) COMMENT '긍부정판단(모델링)',
                `keywords`        varchar(255) COMMENT '키워드 추출(모델링)',
                `search_caterory` varchar(255) COMMENT '검색 카테고리',
                `survey`          text COMMENT '쿠팡 자체 리뷰 설문조사 내용',
                PRIMARY KEY (`review_id`)
            ) COMMENT = '리뷰 정보 테이블'
        """)


def create_conn():
    config = configparser.ConfigParser()
    config.read(f'../../config/config-{env}.ini')  # 환경에 맞는 설정 파일 읽기
    mysql_config = config['mysql']
    return pymysql.connect(host=mysql_config['host'], user=mysql_config['user'], password=mysql_config['password'],
                           db=mysql_config['db'], charset=mysql_config['charset'])


def insert_product_data(csv_file, conn, cursor, version):
    with open(csv_file, 'r', encoding='utf-8') as f_products:
        csvReader = csv.reader(f_products)
        next(csvReader)  # 헤더 건너뛰기
        for row in tqdm(csvReader):
            search_name, unique_product_id, top_cnt, prod_name, price, review_cnt, avg_rating, ad_yn, url = row[0], row[1], row[2], row[3], row[4], \
                row[5], row[6], row[7], row[8]
            sql = f"""
                    INSERT INTO products_{version}
                    (search_name, prod_name, price, review_cnt, avg_rating, ad_yn, url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
            cursor.execute(sql, (search_name, prod_name, price, 0, avg_rating, 'Y', url))
            conn.commit()


def insert_review_data(csv_file, conn, cursor, version):
    with open(csv_file, 'r', encoding='utf-8') as f_reviews:
        csvReader = csv.reader(f_reviews)
        next(csvReader)  # 헤더 건너뛰기
        for row in tqdm(csvReader):
            prod_name, user_name, rating, title, context, answer, helped_cnt, top100_yn \
                = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
            cursor.execute(f"SELECT product_id FROM products_{version} WHERE prod_name = %s", (prod_name,))
            result = cursor.fetchone()
            if result is None:
                print(f"No product found for name: {prod_name}")
                continue
            product_id = result[0]
            sql = f"""
                    INSERT INTO reviews_{version}
                    (prod_id, prod_name, rating, title, context, answer, helped_cnt, top100_yn)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
            cursor.execute(sql, (product_id, prod_name, rating, title, context, answer, helped_cnt, top100_yn))
            conn.commit()


def run_pipeline(product_csv_file, review_csv_file, version):
    # MySQL 연결 설정
    conn = create_conn()
    curs = conn.cursor()

    # 테이블 생성
    print("Creating tables...")
    check_products_table(curs, version)
    check_reviews_table(curs, version)

    # JSON_FILE : str = './json/headers.json'
    # JSON_FILE = Path(__file__).resolve().parent
    # product_csv_path = os.path.join(JSON_FILE, 'crawler', f'{product_csv_file}.csv')
    # review_csv_path = os.path.join(JSON_FILE, 'crawler', f'{review_csv_file}.csv')

    insert_product_data(product_csv_file, conn, curs, version)
    insert_review_data(review_csv_file, conn, curs, version)

    # 제품 데이터 삽입
    # print("Inserting product data...")
    # insert_product_data(f'../crawler/{product_csv_file}.csv', conn, curs, version)
    #
    # # 리뷰 데이터 삽입
    # print("Inserting review data...")
    # insert_review_data(f'../crawler/{review_csv_file}.csv', conn, curs, version)

    # 연결 종료
    conn.close()
    print("All Done!")


if __name__ == "__main__":
    pass
    # product_csv_file = "products_ver4"
    # review_csv_file = "review_products_ver4"
    # version = 'ver8'
    #
    # current_directory = Path(__file__).resolve().parent.parent.parent
    # print(current_directory)
    # product_csv_path = current_directory.joinpath("utils", "crawler", f"{product_csv_file}.csv")
    # review_csv_path = current_directory.joinpath("utils", "crawler", f"{review_csv_file}.csv")
    #
    # run_pipeline(product_csv_path, review_csv_path, version)
