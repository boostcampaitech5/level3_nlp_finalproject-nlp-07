import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def crawling_products(search_list):
    data = []
    total_size = sum(len(v) for v in search_list.values())
    cnt = 1

    for search_category, search_items in search_list.items():
        for search_item in search_items:
            print(search_item)
            url = os.getenv('CRAWL_ENDPOINT')
            path = url + '/crawling/' + search_item


            # url = f"https://www.coupang.com/np/search?q={search_item}&channel=user&component=&eventCategory=SRP&trcid=&traid=&sorter=scoreDesc&minPrice=&maxPrice=&priceRange=&filterType=&listSize=36&filter=&isPriceRange=false&brand=&offerCondition=&rating=0&page=1&rocketAll=false&searchIndexingToken=1=6&backgroundColor="

            # headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36", "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3"}
            res = requests.get(path).json()
            # res.raise_for_status()

            soup = BeautifulSoup(res['text'], 'lxml')
                
            products = soup.select('.search-product')

            rank_list = ['no-1', 'no-2', 'no-3', 'no-4', 'no-5', 'no-6',
                    'no-7', 'no-8', 'no-9', 'no-10']
            

            unique_product_ids = []
            for product in products:
                try:
                    unique_product_id = product['data-product-id']

                    # 중복 상품 제거
                    if unique_product_id in unique_product_ids:
                        continue
                    

                    # 먼저 None인지 검사하고, 이후 attrs['class'] 접근
                    number_elem = product.select_one('.number')
                    if number_elem is None: # 랭킹 없는 상품 먼저 거른다!
                        # print("None")
                        continue
                    elif number_elem.attrs['class'][1] not in rank_list: # 랭킹 없는 상품 먼저 거른다!
                        # print("not in rank_list")
                        continue

                    if number_elem.attrs['class'][1] == 'no-10':
                        break
                    
                    top_cnt = product.select_one('.number').attrs['class'][0]+" "+product.select_one('.number').attrs['class'][1]+" "
                    name = product.select_one('.name').get_text()
                    name = search_item+" "+name
                    price = product.select_one('.price-value').get_text().replace(',', '')
                    review_cnt = product.select_one('.rating-total-count')
                    review_cnt = review_cnt.get_text().replace('(', '').replace(')', '') if review_cnt else '0'
                    rating = product.select_one('.rating')
                    rating = rating.get_text() if rating else '0.0'
                    ad_yn = 'Y' if product.select_one('.ad-badge') else 'N'
                    url = "https://www.coupang.com" + product.select_one('a.search-product-link')['href']
                    product_img_url = "https:"+product.select_one('.search-product-wrap-img')['src']
                    # print(product_img_url)
                    


                    if 'blank' in product_img_url:
                        # print("blank")
                        # print(cnt)
                        product_img_url = "https:"+product.select_one('.search-product-wrap-img')['data-img-src']

                    data.append([search_item, unique_product_id, top_cnt, name, price, review_cnt, rating, ad_yn, url, product_img_url])
                    unique_product_ids.append(unique_product_id)

                    cnt += 1
                except Exception as e:
                    print(f"Error message: {e}")
                    continue
            
            # 랭킹 없는 것을 임의로 가져와서 10개 채우도록 함.
            if len(data) < 10:               
                print("랭킹 없는 것을 임의로 가져와서 10개 채우도록 함.")
                for product in products:                    
                    try:
                        unique_product_id = product['data-product-id']

                        # 중복 상품 제거
                        if unique_product_id in unique_product_ids:
                            continue
                                                
                        top_cnt = "top-out" # 랭킹 없는 것

                        name = product.select_one('.name').get_text()
                        name = search_item+" "+name
                        price = product.select_one('.price-value').get_text().replace(',', '')
                        review_cnt = product.select_one('.rating-total-count')
                        review_cnt = review_cnt.get_text().replace('(', '').replace(')', '') if review_cnt else '0'
                        rating = product.select_one('.rating')
                        rating = rating.get_text() if rating else '0.0'
                        ad_yn = 'Y' if product.select_one('.ad-badge') else 'N'
                        url = "https://www.coupang.com" + product.select_one('a.search-product-link')['href']
                        product_img_url = "https:"+product.select_one('.search-product-wrap-img')['src']
                        # print(product_img_url)

                        if 'blank' in product_img_url:
                            # print("blank")
                            # print(cnt)
                            product_img_url = "https:"+product.select_one('.search-product-wrap-img')['data-img-src']

                        data.append([search_item, unique_product_id, top_cnt, name, price, review_cnt, rating, ad_yn, url, product_img_url])
                        unique_product_ids.append(unique_product_id)

                        cnt += 1

                        if len(data) == 10:
                            break


                    except Exception as e:
                        print(f"Error message: {e}")
                        continue


        df = pd.DataFrame(data, columns=['search_name', 'unique_product_id', 'top_cnt', 'name', 'price', 'review_cnt', 'rating', 'ad_yn', 'URL', 'product_img_url'])
        df.to_csv(f"{search_item}_bs4.csv", index=False)

        # print(df)

        return f"{search_item}_bs4"

if __name__ == '__main__':
    search_list = {'노트북': ['감']}
    crawling_products(search_list)
    print("All Done!")
