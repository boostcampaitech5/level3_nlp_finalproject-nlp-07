from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from keybert import KeyBERT


reviews = pd.read_csv('../utils/db_scripts/output.csv')['context'][:10]



keywords = []
for review in reviews:
    # 키워드 추출
    kw_model = KeyBERT()
    extracted_keywords = kw_model.extract_keywords(review)
    keywords.extend(extracted_keywords)

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
model = AutoModel.from_pretrained('klue/roberta-large')

vectors = []

from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
model = AutoModel.from_pretrained('klue/roberta-large')

vectors = []

# 키워드 벡터화
for keyword, _ in extracted_keywords:
    inputs = tokenizer(keyword, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    vectors.append(outputs.last_hidden_state.mean(dim=1).detach().numpy().squeeze())

print(vectors)

# 클러스터링
vectors = np.stack(vectors)  # Convert list of arrays into a single 2D array
kmeans = KMeans(n_clusters=2, random_state=0).fit(vectors)

# 클러스터 결과 출력
for idx, label in enumerate(set(kmeans.labels_)):
    print(f'Cluster {idx+1}:')
    print([keywords[i][0] for i in range(len(kmeans.labels_)) if kmeans.labels_[i]==label])
    print()



