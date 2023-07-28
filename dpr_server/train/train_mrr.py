from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm, trange
import random
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    BertConfig,
)
import transformers
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls,
        cfg_name: str = "klue/bert-base",
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        cfg = BertConfig.from_pretrained(cfg_name)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        representation_token_pos=0,
    ):
        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        # return sequence_output, pooled_output, hidden_states
        return pooled_output


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


def train(args, train_dataset, p_model, q_model, test_dataset):
    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
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

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    min_val_loss = 2e9
    max_MRR = 0
    for epoch in train_iterator:
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
        print(f"\nEPOCH {epoch+1} Done!")

        q_model.eval()
        p_model.eval()
        total_r_score = 0
        top1_hit = 0
        top3_hit = 0
        top5_hit = 0
        for batch in test_dataset:
            q_seqs_val = batch[0]
            p_seqs_val = batch[1]

            q_emb = q_model(**q_seqs_val)
            p_emb = p_model(**p_seqs_val)
            sim_scores = torch.matmul(p_emb, torch.transpose(q_emb, 0, 1)).squeeze()
            sorted_scores = torch.argsort(sim_scores).tolist()
            answer_index = sorted_scores.index(0) + 1

            if answer_index == 1:
                top1_hit += 1
            if answer_index <= 3:
                top3_hit += 1
            if answer_index <= 5:
                top5_hit += 1
            total_r_score += 1 / answer_index

        n = len(test_data)
        MRR = total_r_score / n
        TOP1_hit = top1_hit / n
        TOP3_hit = top3_hit / n
        TOP5_hit = top5_hit / n
        print(f"MRR     : {MRR:0.4f}")
        print(f"TOP1 hit: {TOP1_hit:0.4f}")
        print(f"TOP3 hit: {TOP3_hit:0.4f}")
        print(f"TOP5 hit: {TOP5_hit:0.4f}\n")

        if MRR > max_MRR:
            max_MRR = MRR
            # p_model.save_pretrained("./bert/p_encoder")
            # q_model.save_pretrained("./bert/q_encoder")
            print(f"New checkpoint saved.")


if __name__ == "__main__":
    model_checkpoint = "klue/bert-base"
    # model_checkpoint = "beomi/kcbert-base"
    os.makedirs("./bert", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    data = pd.read_csv("./testset_40.csv")
    dataset_train, dataset_valid = train_test_split(
        data, test_size=0.1, random_state=42
    )
    with open("./test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    query, context = [], []
    for v in test_data.values():
        query.append(v["query"])
        context.append(v["positive_passage"])
    # query = list(dict.fromkeys([v["query"] for v in test_data.values()]))
    # context = list(dict.fromkeys([v["positive_passage"] for v in test_data.values()]))

    print("Tokenizing ...")
    train_q_seqs = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    train_p_seqs = tokenizer(
        context,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # train_q_seqs = tokenizer(
    #     list(dataset_train["query"]),
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # train_p_seqs = tokenizer(
    #     list(dataset_train["positive_context"]),
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    # )

    train_dataset = TensorDataset(
        train_p_seqs["input_ids"],
        train_p_seqs["attention_mask"],
        train_p_seqs["token_type_ids"],
        train_q_seqs["input_ids"],
        train_q_seqs["attention_mask"],
        train_q_seqs["token_type_ids"],
    )

    # valid_q_seqs = tokenizer(
    #     list(dataset_valid["query"]),
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # valid_p_seqs = tokenizer(
    #     list(dataset_valid["positive_context"]),
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # valid_dataset = TensorDataset(
    #     valid_p_seqs["input_ids"],
    #     valid_p_seqs["attention_mask"],
    #     valid_p_seqs["token_type_ids"],
    #     valid_q_seqs["input_ids"],
    #     valid_q_seqs["attention_mask"],
    #     valid_q_seqs["token_type_ids"],
    # )
    test_dataset = []
    for key in test_data:
        query = test_data[key]["query"]
        reviews = test_data[key]["total_passages"]
        q_seqs_val = tokenizer(
            [query], padding="max_length", truncation=True, return_tensors="pt"
        )
        q_seqs_val = {
            "input_ids": q_seqs_val["input_ids"].cuda(),
            "attention_mask": q_seqs_val["attention_mask"].cuda(),
            "token_type_ids": q_seqs_val["token_type_ids"].cuda(),
        }
        p_seqs_val = tokenizer(
            reviews, padding="max_length", truncation=True, return_tensors="pt"
        )
        p_seqs_val = {
            "input_ids": p_seqs_val["input_ids"].cuda(),
            "attention_mask": p_seqs_val["attention_mask"].cuda(),
            "token_type_ids": p_seqs_val["token_type_ids"].cuda(),
        }
        test_dataset.append([q_seqs_val, p_seqs_val])

    print("Data Loaded")

    # p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    # q_encoder = BertEncoder.from_pretrained(model_checkpoint)
    p_encoder = HFBertEncoder.init_encoder(cfg_name="JLake310/bert-p-encoder").to(
        "cuda"
    )
    q_encoder = HFBertEncoder.init_encoder(cfg_name="JLake310/bert-q-encoder").to(
        "cuda"
    )
    print("Model Loaded")

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        weight_decay=0.01,
    )

    print("Train Start")
    train(args, train_dataset, p_encoder, q_encoder, test_dataset)
