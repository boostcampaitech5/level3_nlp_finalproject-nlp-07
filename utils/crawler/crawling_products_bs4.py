import requests
from bs4 import BeautifulSoup
import pandas as pd

def crawling_products(search_list):
    data = []
    total_size = sum(len(v) for v in search_list.values())
    cnt = 1

    for search_item in search_list.values():
        url = f"https://www.coupang.com/np/search?q={search_item}&channel=user&component=&eventCategory=SRP&trcid=&traid=&sorter=scoreDesc&minPrice=&maxPrice=&priceRange=&filterType=&listSize=36&filter=&isPriceRange=false&brand=&offerCondition=&rating=0&page=1&rocketAll=false&searchIndexingToken=1=6&backgroundColor="

        headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36", "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, 'lxml')
            
        products = soup.select('.search-product')

        rank_list = ['number no-1 ', 'number no-2 ', 'number no-3 ', 'number no-4 ', 'number no-5 ', 'number no-6 ',
                 'number no-7 ', 'number no-8 ', 'number no-9 ', 'number no-10 ']
        


        for product in products:
            try:
                unique_product_id = product['data-product-id']

                if product.select_one('.number').attrs['class'][1] in rank_list:
                    continue
                
                top_cnt = f"number no-{cnt}"
                name = product.select_one('.name').get_text()
                price = product.select_one('.price-value').get_text().replace(',', '')
                review_cnt = product.select_one('.rating-total-count')
                review_cnt = review_cnt.get_text().replace('(', '').replace(')', '') if review_cnt else '0'
                rating = product.select_one('.rating')
                rating = rating.get_text() if rating else '0.0'
                ad_yn = 'Y' if product.select_one('.ad-badge') else 'N'
                url = "https://www.coupang.com" + product.select_one('a.search-product-link')['href']
                product_img_url = product.select_one('.search-product-wrap-img')['src']

                data.append([search_item, unique_product_id, top_cnt, name, price, review_cnt, rating, ad_yn, url, product_img_url])
                cnt += 1
            except Exception as e:
                print(f"Error message: {e}")
                continue

        df = pd.DataFrame(data, columns=['search_name', 'unique_product_id', 'top_cnt', 'name', 'price', 'review_cnt', 'rating', 'ad_yn', 'URL', 'product_img_url'])
        df.to_csv(f"{search_item}_bs4.csv", index=False)

        print(df)

if __name__ == '__main__':
    search_list = {'노트북': ['감']}
    crawling_products(search_list)
    print("All Done!")
