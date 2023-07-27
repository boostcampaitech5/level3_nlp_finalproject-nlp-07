import torch
from torch import nn

from transformers import (
    BertModel,
    BertPreTrainedModel,
    RobertaModel,
    RobertaPreTrainedModel,
    BartModel,
    BartPretrainedModel,
    T5Model,
    T5PreTrainedModel,
    ElectraPreTrainedModel,
    ElectraModel,
    AutoModel,
    BertConfig,
)

import transformers


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
