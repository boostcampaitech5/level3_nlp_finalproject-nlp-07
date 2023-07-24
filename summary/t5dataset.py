import torch
from torch.utils.data import Dataset
import random
import string

class T5Dataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['filtered_context']
        instruction_text = data['instruction']
        targets = data['summary']
        self.inputs = []
        self.labels = []
        for text,instruction_text ,label in zip(input_texts,instruction_text,targets):
            tokenized_input = tokenizer(text+','+instruction_text, padding='max_length', truncation=True, return_tensors='pt',max_length=1024)
            tokenized_target = tokenizer(label, padding='max_length', truncation=True, return_tensors='pt',max_length=1024)
            self.inputs.append(tokenized_input)
            self.labels.append(tokenized_target['input_ids'])
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    def __len__(self):
        return len(self.labels)