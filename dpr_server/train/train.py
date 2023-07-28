from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm, trange
import random
import torch
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import pickle

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


def train(args, train_dataset, valid_dataset, p_model, q_model, save_dir):
    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=2,
    )

    ### 추가 부분 ###
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in p_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in q_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    ### 추가 부분 ###

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Start training!
    global_step = 0

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()
    p_encoder = p_model
    q_encoder = q_model

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    min_val_loss = 2e9
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            q_model.train()
            p_model.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
            q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_outputs, 0, 1)
            )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, len(batch[0])).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(sim_scores, dim=1)

            try:
                loss = F.nll_loss(sim_scores, targets)
                epoch_iterator.set_postfix(loss=loss.item())
            except:
                print("continue")
                continue

            # Softmax를 취한 후에 loss를 계산하므로 digonal한 요소를 제외한 나머지 연산값은 자연스레 떨어지는 효과

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()

        # epoch done, Validation
        epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
        val_loss = 0
        for step, batch in enumerate(epoch_iterator):
            q_model.eval()
            p_model.eval()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
            q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_outputs, 0, 1)
            )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, len(batch[0])).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(sim_scores, dim=1)

            try:
                loss = F.nll_loss(sim_scores, targets)
                epoch_iterator.set_postfix(loss=loss.item())
            except:
                print("continue")
                continue
            val_loss += loss.item()
            # print(loss.item())

            torch.cuda.empty_cache()

        print(f"\nval_loss : {val_loss:0.5f}\n")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            p_encoder.save_pretrained(os.path.join(save_dir, "p_encoder"))
            q_encoder.save_pretrained(os.path.join(save_dir, "q_encoder"))
            p_encoder = p_model
            q_encoder = q_model
            print(f"New checkpoint saved.")

    return p_encoder, q_encoder


def test(p_encoder, q_encoder, tokenizer):
    p_encoder.eval()
    q_encoder.eval()

    p_embs = []
    summary_test_emb_path = "./data/summary_test_emb.bin"
    test_set = pd.read_csv("./data/summary_test.csv")
    if os.path.isfile(summary_test_emb_path):
        with open(summary_test_emb_path, "rb") as file:
            p_embs = pickle.load(file)
    else:
        for p in test_set["context"]:
            p = tokenizer(
                p, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            p_emb = p_encoder(**p).cpu().detach().numpy()
            p_embs.append(p_emb)
            with open(summary_test_emb_path, "wb") as f:
                pickle.dump(p_embs, f)

    top1_hit = 0
    top3_hit = 0
    top5_hit = 0
    total_r_score = 0
    p_embs = np.array(p_embs)
    for q_id, query in enumerate(test_set["query"]):
        q_seqs_val = tokenizer(
            [query], padding="max_length", truncation=True, return_tensors="pt"
        ).to("cuda")
        q_emb = q_encoder(**q_seqs_val).cpu().detach()  # (num_query, emb_dim)
        p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze().tolist()
        answer_index = rank.index(q_id) + 1

        if answer_index == 1:
            top1_hit += 1
        if answer_index <= 3:
            top3_hit += 1
        if answer_index <= 5:
            top5_hit += 1
        total_r_score += 1 / answer_index

    n = len(test_set)
    MRR = total_r_score / n
    TOP1_hit = top1_hit / n
    TOP3_hit = top3_hit / n
    TOP5_hit = top5_hit / n

    print(f"MRR     : {MRR:0.4f}")
    print(f"TOP1 hit: {TOP1_hit:0.4f}")
    print(f"TOP3 hit: {TOP3_hit:0.4f}")
    print(f"TOP5 hit: {TOP5_hit:0.4f}")

    return MRR, TOP1_hit, TOP3_hit, TOP5_hit


if __name__ == "__main__":
    model_checkpoint = "klue/bert-base"
    # model_checkpoint = "beomi/kcbert-base"
    save_dir = "./summary_data"
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    dataset_train = pd.read_csv("./data/summary_train.csv")
    dataset_valid = pd.read_csv("./data/summary_val.csv")

    print("Tokenizing ...")
    train_q_seqs = tokenizer(
        list(dataset_train["query"]),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    train_p_seqs = tokenizer(
        list(dataset_train["context"]),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    train_dataset = TensorDataset(
        train_p_seqs["input_ids"],
        train_p_seqs["attention_mask"],
        train_p_seqs["token_type_ids"],
        train_q_seqs["input_ids"],
        train_q_seqs["attention_mask"],
        train_q_seqs["token_type_ids"],
    )

    valid_q_seqs = tokenizer(
        list(dataset_valid["query"]),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    valid_p_seqs = tokenizer(
        list(dataset_valid["context"]),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    valid_dataset = TensorDataset(
        valid_p_seqs["input_ids"],
        valid_p_seqs["attention_mask"],
        valid_p_seqs["token_type_ids"],
        valid_q_seqs["input_ids"],
        valid_q_seqs["attention_mask"],
        valid_q_seqs["token_type_ids"],
    )
    print("Data Loaded")

    p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint)
    print("Model Loaded")

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        num_train_epochs=20,
        weight_decay=0.01,
    )

    print("Train Start")
    p_encoder, q_encoder = train(
        args, train_dataset, valid_dataset, p_encoder, q_encoder, save_dir
    )

    print("Test Start")
    MRR, TOP1_hit, TOP3_hit, TOP5_hit = test(p_encoder, q_encoder, tokenizer)
