from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig, default_data_collator
from datasets import load_dataset
import torch

from utils.prompter import Prompter

MODEL = "EleutherAI/polyglot-ko-5.8b"
DEVICE = torch.device("cuda")

config = AutoConfig.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    config = config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map = "auto"
)

train_dataset = load_dataset("boostcamp-5th-nlp07/review_summary_v1", split="train")
prompter = Prompter("/opt/ml/input/level3_nlp_finalproject-nlp-07/summary/templates/summary_v1.0_train.json")

def preprocess(example):
   input_text = [prompter.generate_prompt(review=rev, summary=summ) for rev, summ in zip(example["review"], example["summary"])]
   outputs = tokenizer(input_text, max_length=256, truncation=True, 
                       padding=True
                       )
   outputs["labels"] = outputs["input_ids"].copy()
   return outputs

tokenized_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # dynamic padding

training_args = TrainingArguments(
    output_dir = "polyglot",
    optim="adafactor", # 메모리 사용 감소
    num_train_epochs=5,
    per_device_train_batch_size=4,
    fp16=True,
    learning_rate=2e-5,
    logging_strategy="steps",
    logging_steps=0.1,
    evaluation_strategy="no",
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

trainer.train()

trainer.save_model()