import sys
import os
sys.path.append("/opt/ml/input/level3_nlp_finalproject-nlp-07/") # 프로젝트 경로 수정
from utils.preprocess import clean_text
from filtering.review_filter import ReviewFilter

import pandas as pd
from tqdm import tqdm

INPUT_CSV_PATH = "/opt/ml/input/data/reviews_ver31.csv"
filename, ext = os.path.splitext(INPUT_CSV_PATH)
OUTPUT_CSV_PATH = filename + "_clean"+ext # {filename}_clean.csv

print("Input file:", INPUT_CSV_PATH)
print("Output file:", OUTPUT_CSV_PATH)

filter = ReviewFilter()

df = pd.read_csv(INPUT_CSV_PATH)

tqdm.pandas(desc="Clean and filter context")
df["context"] = df["context"].progress_apply(lambda x: 
    filter.filter_text(clean_text(x, remove_dot=False))
    )

df.to_csv(OUTPUT_CSV_PATH)


