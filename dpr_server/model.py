import torch
import torch.nn.functional as F
from encoder import HFBertEncoder
from typing import Callable, Dict, List, Tuple
from transformers import AutoTokenizer


class DenseRetriever:
    def __init__(self):
        self.p_encoder = HFBertEncoder.init_encoder(
            cfg_name="JLake310/bert-p-encoder"
        ).eval()
        self.q_encoder = HFBertEncoder.init_encoder(
            cfg_name="JLake310/bert-q-encoder"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("JLake310/bert-p-encoder")

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
        q_seqs_val = self.tokenizer(
            [query], padding="max_length", truncation=True, return_tensors="pt"
        )
        q_emb = self.q_encoder(**q_seqs_val)

        max_val = -999
        max_idx = 0
        for idx, review in enumerate(reviews):
            r_tokens = self.tokenizer(
                review,
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

        return reviews[max_idx]
