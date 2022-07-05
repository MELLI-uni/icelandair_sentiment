import pandas as pd

import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer, RobertaModel
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import logging

device = 'cuda' if cuda.is_available() else 'cpu'

logging.set_verbosity_warning()
logging.set_verbosity_error()

# Load model and tokenizer
model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

df = pd.read_pickle('../Data/Tuning/tuning_eng.pkl')
df.to_csv('train_text.txt', header=None, index=None, sep=' ', mode='w')
#print(df)

dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path='./train_text.txt',
        block_size=512,
        )

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        )

training_args = TrainingArguments(
        output_dir='./roberta-retrained',
        overwrite_output_dir=True,
        num_train_epochs=25,
        per_device_train_batch_size=48,
        save_steps=500,
        save_total_limit=2,
        seed=1,
        )

trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        )

trainer.train()
trainer.save_model('./roberta-retrained')
