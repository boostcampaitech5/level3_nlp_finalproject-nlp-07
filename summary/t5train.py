import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer,AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from t5dataset import T5Dataset
from datasets import load_dataset, load_metric
def main():
    # Load pre-trained model and tokenizer
    model_name = "lcw99/t5-large-korean-text-summary" 
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #, torch_dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    data = pd.read_csv(os.path.join(DATA_DIR, 'train_final.csv'))
    dataset_train, dataset_valid = train_test_split(data, test_size=0.1, random_state=42)
    dataset_valid.to_csv('valid_final.csv')
    train_dataset = T5Dataset(dataset_train,tokenizer)  # Your training dataset
    eval_dataset = T5Dataset(dataset_valid,tokenizer) # Your evaluation dataset
    data_collator = DataCollatorForSeq2Seq(tokenizer,model=model)
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models",
        num_train_epochs=5,
        # max_steps=900,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        evaluation_strategy="epoch", #step학습 시 steps로 변경
        save_steps=200,
        save_strategy="epoch", #step학습 시 steps로 변경
        logging_dir="./logs",
        #fp16=True,
        save_total_limit = 2,
        weight_decay=0.01,
        logging_steps = 50,
        dataloader_num_workers=0,
        gradient_accumulation_steps = 8, #batch
    ) 
    # Define the fine-tuning trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator = data_collator,
    )

    
    # Fine-tune the model
    trainer.train()
if __name__ == "__main__":
    main()