import sys
import pickle
import json

import regex as re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tabulate import tabulate

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
#   4. Icelandic modal verb list
#   5. English degree adverbs dictionary
#   6. Icelandic degree adverbs dictionary
with open('dictionaries.pickle', 'rb') as handle:
    dictionaries = pickle.load(handle)

emoji_dict = dictionaries[1]
flight_dict = dictionaries[2]
isk_stop = dictionaries[3]
isk_modal = dictionaries[4]
isk_deg = dictionaries[6]

handle.close()

with open('posmap.pickle', 'rb') as handle:
    tagmap = pickle.load(handle)
handle.close()

NEUTRAL_SKIP = ["N/A", "n/a", "na", "N/a", "n/A", "NA", "nei", "Nei"]

def data_cleaning(df):
    # Remove na values from dataframe
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Remove duplicate values from dataframe
    df = df.drop_duplicates()

    sentences = []

    for line in df.values:
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
                        tokens.append(item.txt.lower())

                sentences.append(tuple(tokens))

    return sentences

def tag_n_lemmatize(tokenized_inputs): 
    tags = tagger.tag_bulk(
        tokenized_inputs, 
        batch_size=2
    )

    tags, lemmas = Lemmatize(tokenized_inputs, tags)

    return tags, lemmas

def train_word2vec(tokenized_inputs):
    skip, lemmas = tag_n_lemmatize(tokenized_inputs)

    sent_corpus = []

    for sent in lemmas:
        temp = []
        for token in sent:
            if token.isalpha() == False:
                continue
            elif (
                    token in isk_stop or 
                    token in isk_deg
                ):
                continue
            else:
                temp.append(token)

        sent_corpus.append(temp)

    print(sent_corpus)

    model = Word2Vec(sentences=sent_corpus, vector_size=200, window=4, min_count=1, workers=4)
    model.save("icelandic_word2vec.model")

    return

def filtering(lemma_list, tag_list, sentiment):
    lemma_mod = []
    tag_mod = []
    senti_mod = []

    if sentiment.lower() == "positive":
        score = 1
    elif sentiment.lower() == "negative" or sentiment.lower() == 'negativa':
        score = -1
    else:
        score = 0

    negated = False

    for i in range(len(tag_list)):
        if lemma_list[i] == "ekki":
            negated = True
            continue

        if lemma_list[i] in isk_deg:
            continue

        if lemma_list[i] in isk_stop:
            continue

        if tag_list[i] in tagmap:
            gen_tag = tagmap[tag_list[i]]
            if gen_tag == "ADJ" or gen_tag == "ADV":
                tag_mod.append(tagmap[tag_list[i]])
                lemma_mod.append(lemma_list[i])
                if negated == True:
                    senti_mod.append(-score)
                    negated = False
                else: 
                    senti_mod.append(score)

    return lemma_mod, tag_mod, senti_mod

def process_dataframe(df):
    df.reset_index(inplace=True, drop=True)

    df_text = df.copy()
    del df_text['Sentiment']

    sentences = data_cleaning(df_text)
    tags, lemmas = tag_n_lemmatize(tuple(sentences))

    del df['answer_freetext_value']
    df = df.join(pd.Series(lemmas, name='tmp_lemma'))
    df = df.join(pd.Series(tags, name='tmp_tag'))

    df = df.apply(lambda x: filtering(x['tmp_lemma'], x['tmp_tag'], x['Sentiment']), axis=1, result_type='expand')
    df.columns = ['answer_freetext_value', 'tag', 'Sentiment']

    del df['tag']
    
    return df

def open_lexicon(file_name):
    path = './'

    lexicon = {}
    tuning = {}

    with open((path+file_name), encoding='utf-8') as f:
        for line in f:
            word, mean, scores = line.split("\t")
            lexicon[word] = float(mean.strip())
            tuning[word] = [int(x) for x in scores.strip('\n[]').split(', ')]

    f.close()

    return lexicon, tuning

def update_lexicon(df):
    path = './'
    file_name = 'isk_lexicon.txt'

    lexicon, tuning = open_lexicon(file_name)

    df = df.explode(['answer_freetext_value', 'Sentiment'], ignore_index=True)
    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    f = open((path+file_name), 'w', encoding='utf-8')

    for row in df.itertuples(index=False):
        new_word = row.answer_freetext_value
        new_score = row.Sentiment

        if new_score == 1:
            place = 0
        elif new_score == 0:
            place = 1
        else:
            place = 2

        if new_word in tuning:
            tuning[new_word][place] += 1
        else:
            tuning[new_word] = [0, 0, 0]
            tuning[new_word][place] += 1
        
    tuning = sorted(tuning.items())

    for item in tuning:
        mean = ((item[1][0] * 1) + (item[1][2] * -1)) / sum(item[1])
        f.write(item[0] + "\t" + str(mean) + "\t" + str(item[1]) + "\n")

    f.close()

df_train = pd.read_pickle('./isk_train.pkl')
del df_train['id']
df_test = pd.read_pickle('./isk_test.pkl')
del df_test['id']
df_unlabeled = pd.read_pickle('./tuning_isk.pkl')

#cleaned_df = process_dataframe(df_train)
#update_lexicon(cleaned_df)

#test_lexicon(df_test)

cleaned_df = process_dataframe(df_train)
update_lexicon(cleaned_df)
#cleaned_df.to_csv('checkpoint.txt', header=None, index=None, sep=' ', mode='w')

