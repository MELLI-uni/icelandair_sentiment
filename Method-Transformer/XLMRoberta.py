from transformers import pipeline
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")

dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path='./tuning_isk.txt',
        block_size=512,
        )

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        )

training_args = TrainingArguments(
        output_dir='./xlmIS-retrained',
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
        train_dataset=dataset
        )

trainer.train()
trainer.save_model("./xlmIS-retrained")

#fill_mask = pipeline(
#        "fill-mask",
#        model="./roberta-retrained",
#        tokenizer="roberta-base",
#        )

#print(fill_mask("Send these <mask> back!"))
