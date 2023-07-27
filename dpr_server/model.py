import torch
import torch.nn.functional as F
from encoder import HFBertEncoder, BertEncoder
from typing import Callable, Dict, List, Tuple
from transformers import AutoTokenizer, AutoConfig
import re
from kiwipiepy import Kiwi
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
import time
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def filter_corpus(corpus):
    corpus = [text for text in corpus if text.find("배송") == -1]
    corpus = [text for text in corpus if text.find("도움") == -1]
    corpus = [text for text in corpus if len(text) > 10]
    corpus = [text for text in corpus if len(text) < 200]
    corpus = [text for text in corpus if text.find("년") == -1]
    corpus = [text for text in corpus if text.find("ml") == -1]
    corpus = [text for text in corpus if text.find("날짜") == -1]
    corpus = [text for text in corpus if text.find("22") == -1]
    corpus = [text for text in corpus if text.find("23") == -1]
    corpus = [text for text in corpus if text.find("기한") == -1]
    corpus = [text for text in corpus if text.find("미리") == -1]
    corpus = [text for text in corpus if text.find("감사") == -1]
    return corpus


from pydantic import BaseModel


class Item(BaseModel):
    prod_id: str
    prod_name: str
    context: str


class DenseRetriever:
    def __init__(self):
        model_name = "klue/bert-base"

        self.p_encoder = HFBertEncoder.init_encoder(
            cfg_name="JLake310/bert-p-encoder"
        ).eval()
        self.q_encoder = HFBertEncoder.init_encoder(
            cfg_name="JLake310/bert-q-encoder"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "JLake310/bert-p-encoder", use_fast=True
        )

        # self.p_encoder = BertEncoder.from_pretrained(model_name).eval()
        # self.q_encoder = BertEncoder.from_pretrained(model_name).eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_dpr_concat(self, query: str, reviews: List[str]) -> str:
        q_seqs_val = self.tokenizer(
            [query], padding="max_length", truncation=True, return_tensors="pt"
        )
        q_emb = self.q_encoder(**q_seqs_val)

        p_seqs_val = self.tokenizer(
            reviews, padding="max_length", truncation=True, return_tensors="pt"
        )

        p_emb = self.p_encoder(**p_seqs_val)
        sim_scores = torch.matmul(p_emb, torch.transpose(q_emb, 0, 1))
        max_idx = torch.argmax(sim_scores)

        return reviews[max_idx]

    def run_dpr_split(self, query: str, reviews: List[str]) -> str:
        start = time.time()
        q_seqs_val = self.tokenizer(
            [query], padding="max_length", truncation=True, return_tensors="pt"
        )
        q_emb = self.q_encoder(**q_seqs_val)

        max_val = -999
        max_idx = 0
        for idx, review in enumerate(reviews):
            r_tokens = self.tokenizer(
                review[:3000],
                truncation=True,
                max_length=512,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
                padding="max_length",
                return_tensors="pt",
            )
            r_input = {
                "input_ids": r_tokens["input_ids"],
                "attention_mask": r_tokens["attention_mask"],
                "token_type_ids": r_tokens["token_type_ids"],
            }
            r_emb = self.p_encoder(**r_input)
            sim_scores = torch.matmul(r_emb, torch.transpose(q_emb, 0, 1))
            mean_val = torch.mean(sim_scores)

            if mean_val > max_val:
                max_val = mean_val
                max_idx = idx
        end = time.time()
        print(f"Retrieve 시간 : {end-start:.5f}sec")
        return reviews[max_idx]

    def run_dpr_db(self, query: str, reviews: List[Dict]) -> str:
        start = time.time()
        reviews = [review["context"] for review in reviews]
        reviews = [re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", text) for text in reviews]
        reviews = [re.sub(r"[^가-힣a-zA-Z0-9\n\s]", "", text).strip() for text in reviews]
        kiwi = Kiwi()

        split_texts = []
        for review in reviews:
            sents = kiwi.split_into_sents(review)
            sents = [sent.text for sent in sents]
            split_texts.append(sents)

        corpus = []
        for idx, text in enumerate(split_texts):
            for t in text:
                corpus.append(str(idx) + " " + t)

        corpus = filter_corpus(corpus)

        clusters = ["평가", "조리"]
        # data = clusters + filter_corpus(split_texts[text_id])
        data = clusters + corpus
        n = len(clusters)

        # Vectorizer
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        tokenizer_func = lambda x: self.tokenizer.tokenize(
            x.translate(remove_punct_dict)
        )
        vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1, 2))

        # Feature vectorize
        feature_vect = vectorizer.fit_transform(data)

        # 임의 클러스터로 클러스터 진행
        km_cluster = KMeans(n_clusters=n, max_iter=10000, random_state=0)
        km_cluster.fit(feature_vect[:n])

        # 임의 클러스터로 새로운 클러스터링 초기화
        kmeans_new = KMeans(init=km_cluster.cluster_centers_, n_clusters=n)
        kmeans_new.fit(feature_vect[n:])

        result = [[] for _ in range(n)]
        for idx, label in enumerate(km_cluster.labels_):
            result[label].append(clusters[idx])

        for idx, label in enumerate(kmeans_new.labels_):
            result[label].append(data[idx + n])

        result_dict = {}
        for r in result:
            r_dict = defaultdict(list)
            for text in r[1:]:
                index = text.split(" ")[0]
                t = " ".join(text.split(" ")[1:])
                r_dict[index].append(t)
            # result_dict[r[0]] = r[1:]
            result_dict[r[0]] = dict(r_dict)

        filtered_text = [[] for _ in range(len(split_texts))]

        for idx in result_dict["평가"]:
            filtered_text[int(idx)].extend(result_dict["평가"][idx])

        text_list = [". ".join(text) for text in filtered_text]

        filtered_reviews = []
        for i in range(10):
            filtered_reviews.append(" ".join(text_list[i * 20 : i * 20 + 20]))

        end = time.time()
        print(f"클러스터링 시간 : {end-start:.5f}sec")

        return self.run_dpr_split(query, filtered_reviews)

    def run_dpr_db_v3(self, query: str, reviews: List[Dict]) -> List[Item]:
        prod_name_dict = defaultdict(str)
        for review in reviews:
            if review["prod_id"] not in prod_name_dict:
                prod_name_dict[int(review["prod_id"])] = review["prod_name"]

        prod_ids = [review["prod_id"] for review in reviews]
        contexts = [review["context"] for review in reviews]
        contexts = [re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", text) for text in contexts]
        contexts = [
            re.sub(r"[^가-힣a-zA-Z0-9\n\s]", "", text).strip() for text in contexts
        ]
        kiwi = Kiwi()

        split_texts = []
        for context in contexts:
            sents = kiwi.split_into_sents(context)
            sents = [sent.text for sent in sents]
            split_texts.append(sents)

        corpus = []
        for prod_id, text in zip(prod_ids, split_texts):
            for t in text:
                corpus.append(str(prod_id) + " " + t)

        corpus = filter_corpus(corpus)

        clusters = ["평가", "조리"]
        data = clusters + corpus
        n = len(clusters)

        # Vectorizer
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        tokenizer_func = lambda x: self.tokenizer.tokenize(
            x.translate(remove_punct_dict)
        )
        vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1, 2))

        # Feature vectorize
        feature_vect = vectorizer.fit_transform(data)

        # 임의 클러스터로 클러스터 진행
        km_cluster = KMeans(n_clusters=n, max_iter=10000, random_state=0)
        km_cluster.fit(feature_vect[:n])

        # 임의 클러스터로 새로운 클러스터링 초기화
        kmeans_new = KMeans(init=km_cluster.cluster_centers_, n_clusters=n)
        kmeans_new.fit(feature_vect[n:])

        result = [[] for _ in range(n)]
        for idx, label in enumerate(km_cluster.labels_):
            result[label].append(clusters[idx])

        for idx, label in enumerate(kmeans_new.labels_):
            result[label].append(data[idx + n])

        result_dict = {}
        for r in result:
            r_dict = defaultdict(list)
            for text in r[1:]:
                prod_id = text.split(" ")[0]
                t = " ".join(text.split(" ")[1:])
                r_dict[prod_id].append(t)
            result_dict[r[0]] = dict(r_dict)

        for prod_id, sentence_list in result_dict["평가"].items():
            result_dict["평가"][prod_id] = " ".join(sentence_list)

        filtered_reviews = [item for key, item in result_dict["평가"].items()]
        prod_ids = [key for key, item in result_dict["평가"].items()]

        filtered_reviews

        q_seqs_val = self.tokenizer(
            [query], padding="max_length", truncation=True, return_tensors="pt"
        )
        q_emb = self.q_encoder(**q_seqs_val)

        sim_score_list = []
        for review in filtered_reviews:
            r_tokens = self.tokenizer(
                review[:1000],
                truncation=True,
                max_length=512,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True,
                padding="max_length",
                return_tensors="pt",
            )
            r_input = {
                "input_ids": r_tokens["input_ids"],
                "attention_mask": r_tokens["attention_mask"],
                "token_type_ids": r_tokens["token_type_ids"],
            }
            r_emb = self.p_encoder(**r_input)
            sim_scores = torch.matmul(r_emb, torch.transpose(q_emb, 0, 1))
            mean_val = torch.mean(sim_scores)
            sim_score_list.append(mean_val.tolist())

        sim_score_list = np.array(sim_score_list)
        sort_indices = np.argsort(sim_score_list)

        recom_list = [
            {
                "prod_id": prod_ids[sort_indices[0]],
                "prod_name": prod_name_dict[int(prod_ids[sort_indices[0]])],
                "context": filtered_reviews[sort_indices[0]],
            },
            {
                "prod_id": prod_ids[sort_indices[1]],
                "prod_name": prod_name_dict[int(prod_ids[sort_indices[1]])],
                "context": filtered_reviews[sort_indices[1]],
            },
            {
                "prod_id": prod_ids[sort_indices[2]],
                "prod_name": prod_name_dict[int(prod_ids[sort_indices[2]])],
                "context": filtered_reviews[sort_indices[2]],
            },
        ]

        return recom_list

    def run_dpr_concat_v3(self, query: str, products: List[Dict]) -> List[Item]:
        products = [product for product in products if product["summary"]]
        summary_list = [product["summary"] for product in products]

        q_seqs_val = self.tokenizer(
            [query], padding="max_length", truncation=True, return_tensors="pt"
        )
        q_emb = self.q_encoder(**q_seqs_val)

        p_seqs_val = self.tokenizer(
            summary_list, padding="max_length", truncation=True, return_tensors="pt"
        )

        p_emb = self.p_encoder(**p_seqs_val)
        sim_scores = torch.matmul(p_emb, torch.transpose(q_emb, 0, 1))
        sort_indices = torch.argsort(sim_scores.squeeze())

        return_data = [
            products[sort_indices[0]],
            products[sort_indices[1]],
            products[sort_indices[2]],
        ]

        return return_data
