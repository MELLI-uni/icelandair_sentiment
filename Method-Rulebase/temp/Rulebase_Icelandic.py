import regex as re
import numpy as np
import pickle
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from itertools import islice

from reynir import Greynir
from reynir_correct import tokenize

import torch
import pos

g = Greynir()
device = 'cuda' if cuda.is_available() else 'cpu'

tagger: pos.Tagger = torch.hub.load(
    repo_or_dir="cadia-lvl/POS",
    model="tag",
    device=device,
    force_reload=False,
    force_downloade=False
)

# Load all dictionaries
# Pickle file contains
#   0. English emoji dictionary
#   1. Icelandic emoji dictionary
#   2. Flight and destination list
#   3. Icelandic stopwords list
with open('dictionaries.pickle', 'rb') as handle:
    dictionaries = pickle.load(handle)

emoji_dict = dictionaries[1]
flight_dict = dictionaries[2]
isk_stop = dictionaries[3]

def data_cleaning(df):
    # Remove na values from dataframe
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Remove duplicate values from dataframe
    df = df.drop_duplicates()

    sentences = []

    counter = 0

    for line in df.values:
        counter += 1

        if(counter == 10):
            break

        job = g.submit(line[0])
        for pg in job.paragraphs():
            for sent in pg:
                #sentences.append(sent.tidy_text)
                t = tokenize(str(sent))
                tokens = [item.txt.lower() for item in t if item.txt != '']
                sentences.append(tokens)

    return sentences

df_train = pd.read_pickle('./isk_train.pkl')
df_test = pd.read_pickle('./isk_test.pkl')
df_unlabeled = pd.read_pickle('./tuning_isk.pkl')

tags = tagger.tag_bulk(
    (("), (), batch_size=2)
        )

# sentences_list = data_cleaning(df_unlabeled)
# print(sentences_list)
