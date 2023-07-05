import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import re

# Extract product codes
product_codes = [486586]
outputname='떡볶이'

driver = webdriver.Chrome(executable_path='chromedriver')

data = []

for product_code in product_codes:
    print("product_code: ", product_code)
    url = f"https://www.coupang.com/np/categories/{product_code}"
    # url = 'https://www.coupang.com/np/search?component=&q=%EB%96%A1%EB%B3%B6%EC%9D%B4&channel=user'

    driver.get(url)

    lis = driver.find_elements(By.TAG_NAME, 'li')

    for li in lis:
        try:
            name = li.find_element(By.CLASS_NAME, 'name').text
            price = li.find_element(By.CLASS_NAME, 'price-value').text
            delivery = li.find_element(By.CLASS_NAME, 'delivery').text
            product_url = li.find_element(By.CLASS_NAME, 'baby-product-link').get_attribute('href')
            data.append([name, price, delivery, product_url])
        except Exception:
            pass

driver.quit()

# Convert list of lists into dataframe
df_output = pd.DataFrame(data, columns=["Name", "Price", "Delivery", "URL"])

# Write dataframe to CSV
df_output.to_csv(f"product_{outputname}.csv", index=False)
