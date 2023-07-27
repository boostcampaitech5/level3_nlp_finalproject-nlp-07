from transformers import (
    BertPreTrainedModel,
    BertModel,
    AutoTokenizer,
    ElectraPreTrainedModel,
    ElectraModel,
    BertConfig,
)
import transformers
import torch
from torch import nn
import json
from tqdm import tqdm


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


class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ElectraEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraEncoder, self).__init__(config)

        self.electra = ElectraModel(config)
        self.pooler = ElectraPooler(config)
        self.init_weights

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = self.pooler(outputs[0])

        return pooled_output


if __name__ == "__main__":
    tokenizer_name = "klue/bert-base"
    p_encoder_name = "klue/bert-base"
    q_encoder_name = "klue/bert-base"
    # tokenizer_name = "beomi/kcbert-base"
    # tokenizer_name = "beomi/KcELECTRA-base-v2022"
    # q_encoder_name = "./q_encoder"
    # p_encoder_name = "./p_encoder"
    # q_encoder_name = "beomi/kcbert-base"
    # p_encoder_name = "beomi/kcbert-base"
    # q_encoder_name = "beomi/KcELECTRA-base-v2022"
    # p_encoder_name = "beomi/KcELECTRA-base-v2022"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # q_encoder = BertEncoder.from_pretrained(q_encoder_name).cuda().eval()
    # p_encoder = BertEncoder.from_pretrained(p_encoder_name).cuda().eval()
    p_encoder = HFBertEncoder.init_encoder(cfg_name="JLake310/bert-p-encoder").to(
        "cuda"
    )
    q_encoder = HFBertEncoder.init_encoder(cfg_name="JLake310/bert-q-encoder").to(
        "cuda"
    )
    # q_encoder = ElectraEncoder.from_pretrained(q_encoder_name).eval()
    # p_encoder = ElectraEncoder.from_pretrained(p_encoder_name).eval()

    with open("./test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    top1_hit = 0
    top3_hit = 0
    top5_hit = 0
    total_r_score = 0

    for key in tqdm(test_data):
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
        q_emb = q_encoder(**q_seqs_val)
        p_emb = p_encoder(**p_seqs_val)
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
    print(f"TOP5 hit: {TOP5_hit:0.4f}")
