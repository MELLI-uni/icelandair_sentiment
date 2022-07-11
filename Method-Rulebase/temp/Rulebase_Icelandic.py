import sys
import pickle

import regex as re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from itertools import islice

from reynir import Greynir
from reynir_correct import tokenize

import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import pos

sys.path.append('./nefnir')
from nefnir import Lemmatize

g = Greynir()

tagger: pos.Tagger = torch.hub.load(
    repo_or_dir="cadia-lvl/POS",
    model="tag",
    device=device,
    force_reload=False,
    force_downloader=False
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
                tokens = []
                for item in t:
                    if item.txt == '':
                        continue
                    elif ' ' in item.txt:
                        tokens.extend(re.split(r'\s+', item.txt))
                    else:
                        tokens.append(item.txt)

                sentences.append(tuple(tokens))

    return tuple(sentences)

df_train = pd.read_pickle('./isk_train.pkl')
df_test = pd.read_pickle('./isk_test.pkl')
df_unlabeled = pd.read_pickle('./tuning_isk.pkl')


#sample = (('yndislegt', 'að', 'geta', 'ferðast', 'með', 'ykkur', 'á', 'ný', '.'), ('gott'))

#sent = 'Samskiptafjarlægð þegar nota þarf rútu frá flugstöð að vél er alltof lítil ( og margir í rútunni ) .'

sample = data_cleaning(df_unlabeled)

#print(sample)

tags = tagger.tag_bulk(
    sample, batch_size=2
)  # Batch size works best with GPUs
#print(tags)

#print(parse(sent))

lemmas = Lemmatize(sample, tags)
print(lemmas)

#sentences_list = data_cleaning(df_unlabeled)
#tags = tagger.tag_bulk(
#    sentences_list, batch_size=2
#)

#print(tags)

#for sent in sentences_list:
#    print(sent)
