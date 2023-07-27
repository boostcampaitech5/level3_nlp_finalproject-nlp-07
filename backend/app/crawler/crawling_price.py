import requests
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup as bs
from pathlib import Path
from typing import Optional,Union,Dict,List
from openpyxl import Workbook
import time
import os
import re
import requests as rq
import json
import csv
from datetime import datetime
from pytz import timezone
from tqdm import tqdm
import sys


def get_headers(
    key: str,
    default_value: Optional[str] = None
    )-> Dict[str,Dict[str,str]]:


    """ Get Headers """
    # JSON_FILE : str = './json/headers.json'
    JSON_FILE = Path(__file__).resolve().parent / 'json' / 'headers.json'

    with open(JSON_FILE,'r',encoding='UTF-8') as file:
        headers : Dict[str,Dict[str,str]] = json.loads(file.read())

    try :
        return headers[key]
    except:
        if default_value:
            return default_value
        raise EnvironmentError(f'Set the {key}')
    
def extract_price(url):
    __headers = get_headers(key='headers')
    __headers['referer'] = url 

    # Send a request to the website
    r = requests.get(url, headers=__headers)
    # Parse the HTML document
    soup = BeautifulSoup(r.text, 'html.parser')

    # Assuming price is under a HTML tag with id 'price'
    price = soup.find(class_='total-price').text
    price = re.sub('[^0-9]', '', price)
    price = int(price.replace(',', ''))

    return price


if __name__ == '__main__':

    p = extract_price('https://www.coupang.com/vp/products/166996432?itemId=478240933&vendorItemId=4200250100&pickType=COU_PICK')

    print(p)