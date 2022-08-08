import json
import pandas as pd
import numpy as np

import torch
from torchtext.legacy import data
from torchtext.legacy import datasets

from sklearn.model_selection import train_test_split

df = pd.read_pickle('../Data/eng_total.pkl')
del df['id']

#print(df)

#result = df.to_json(orient='records')

df_train, df_test = train_test_split(df, test_size=0.2, random_state=99)
json_train = df_train.to_json('train.json', orient='records', lines=True)
json_test = df_test.to_json('test.json', orient='records', lines=True)

#print(json_test)

#print(parsed)

TEXT = data.Field(
        tokenize = 'spacy',
        tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField()

#TEXT = data.Field()
#LABEL = data.Field()

fields = {'answer_freetext_value': ('text', TEXT), 'Sentiment': ('label', LABEL)}

#train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)

train_data, test_data = data.TabularDataset.splits(
        path = '.',
        train = 'train.json',
        test = 'test.json',
        format = 'json',
        fields = fields)

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(
        train_data, 
        max_size = MAX_VOCAB_SIZE, 
        vectors = 'glove.6B.100d',
        unk_init = torch.Tensor.normal_)

LABEL.build_vocab(
        train_data)

print(LABEL.vocab.stoi)
