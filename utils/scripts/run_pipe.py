from utils.crawler.crawling_products import crawling_products
from utils.crawler.data.search_list import search_products_list
from utils.crawler.crawling_reviews import Coupang, CSV
from utils.db_scripts.csv2db import run_pipeline
from pathlib import Path

if __name__ == '__main__':
    # 검색할 키워드 가져오기
    search_list = search_products_list()
    # print(search_list)

    # 직접 넣도 싶다면 다음과 같은 형식으로 넣으면 된다.
    search_list = {'음식': ['감']}

    product_file_name = crawling_products(search_list)

    review_file_name = CSV.save_file(product_file_name)

    version = 'ver9'

    current_directory = Path(__file__).resolve().parent.parent.parent
    print(current_directory)
    product_csv_path = current_directory.joinpath("utils", "scripts", f"{product_file_name}.csv")
    review_csv_path = current_directory.joinpath("utils", "scripts", f"{review_file_name}.csv")

    product_csv_file = f"{product_csv_path}"
    review_csv_file = f"{review_csv_path}"

    run_pipeline(product_csv_file, review_csv_file, version)
