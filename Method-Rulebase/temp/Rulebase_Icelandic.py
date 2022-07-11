import sys
import pickle

import regex as re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from itertools import islice
from gensim.models import Word2Vec

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

def tag_n_lemmatize(tokenized_inputs):
    tags = tagger.tag_bulk(
        tokenized_inputs, 
        batch_size=2
    )

    lemmas = Lemmatize(tokenized_inputs, tags)

    return tags, lemmas

def train_word2vec(tokenized_inputs):
    skip, lemmas = tag_n_lemmatize(tokenized_inputs)

    sent_corpus = []

    for sent in lemmas:
        temp = []
        for token in sent:
            if token.isalpha() == False:
                continue
            elif token in isk_stop:
                continue
            else:
                temp.append(token)

        sent_corpus.append(temp)

    print(sent_corpus)

    #model = Word2Vec(sentences=sent_corpus, vector_size=200, window=4, min_count=1, workers=4)
    #model.save("icelandic_word2vec.model")

    return

df_train = pd.read_pickle('./isk_train.pkl')
df_test = pd.read_pickle('./isk_test.pkl')
df_unlabeled = pd.read_pickle('./tuning_isk.pkl')

cleaned_text = data_cleaning(df_unlabeled)
tags, lemmas = tag_n_lemmatize(tuple(cleaned_text))

train_word2vec(tuple(cleaned_text))
