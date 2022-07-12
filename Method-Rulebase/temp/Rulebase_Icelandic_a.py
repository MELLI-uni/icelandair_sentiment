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

    polarity = score

    for i in range(len(tag_list)):
        if lemma_list[i] == 'ekki':
            if lemma_mod and lemma_mod[-1] not in isk_modal:
                senti_mod[-1] *= -1
            else:
                polarity *= -1
            continue

        if tag_list[i] in tagmap:
            tag_mod.append(tagmap[tag_list[i]])
            lemma_mod.append(lemma_list[i])
            senti_mod.append(polarity)

            polarity = score

    return lemma_mod, senti_mod

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
    df.columns = ['answer_freetext_value', 'Sentiment']

    df = df.explode(['answer_freetext_value', 'Sentiment'], ignore_index=True)
    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    return df

def open_lexicon(path, file_name):
    lexicon = {}
    tuning = {}

    with open((path+file_name), 'r', encoding='utf-8') as f:
        for line in f:
            word, mean, scores = line.split("\t")
            lexicon[word] = float(mean.strip())
            tuning[word] = [int(x) for x in scores.strip('\n[]').split(', ')]

    f.close()

    return lexicon, tuning

def find_in_lexicon(tokens, lexicon):
    score = []

    for i in tokens:
        if i in lexicon:
            score.append(lexicon[i])
            continue

        score.append(0)

    return score

def calculate(tokens, lexicon, polarity):
    indiv_score = find_in_lexicon(tokens, lexicon)

    result = np.multiply(indiv_score, polarity)
    score = sum(result)

    if len(tokens) == 1 and tokens[0] in NEUTRAL_SKIP:
        return 'neutral'

    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

def label(df):
    path = './'
    file_name = 'isk_lexicon.txt'

    lexicon, tuning = open_lexicon(path, file_name)

    sentences = data_cleaning(df)
    tags, lemmas = tag_n_lemmatize(tuple(sentences))

    del df['answer_freetext_value']
    df = df.join(pd.Series(lemmas, name='tmp_lemma'))
    df = df.join(pd.Series(tags, name='tmp_tag'))

    df = df.apply(lambda x: filtering(x['tmp_lemma'], x['tmp_tag'], "Positive"), axis=1, result_type='expand')
    df.columns = ['answer_freetext_value', 'Sentiment']

    df_sentiment = df.apply(lambda x: calculate(x['answer_freetext_value'], lexicon, x['Sentiment']), axis=1)
    df['Sentiment'] = df_sentiment

    return df

def accuracy(df_truth, df_predict):
    # Open the two files and convert Sentiment column to list
    senti_truth = df_truth['Sentiment'].tolist()
    labels = np.unique(senti_truth)

    senti_predict = df_predict['Sentiment'].tolist()

    # Calculate precision, recall, accuracy
    precision = precision_score(senti_truth, senti_predict, labels = labels, average = None)
    recall = recall_score(senti_truth, senti_predict, labels = labels, average = None)
    #accuracy = accuracy_score(senti_truth, senti_predict, labels = labels)

    # Calculate f1 score
    f1_gen = f1_score(senti_truth, senti_predict, labels = labels, average = None)
    # Micro average f1 -> calculates positive and negative values globally
    f1_micro = f1_score(senti_truth, senti_predict, labels = labels, average='micro')
    # Macro average f1 -> takes the average of each class's F1 score
    f1_macro = f1_score(senti_truth, senti_predict, labels = labels, average='macro')

    # Compile scores in panda dataframe
    score_compile = np.array([precision, recall, f1_gen])
    f1_average = np.array([f1_micro, f1_macro])
    df_score = pd.DataFrame(data=score_compile, index=['Precision', 'Recall', 'F1'], columns=labels)
    df_average = pd.DataFrame(data=f1_average, index=['F1 Microaverage', 'F1 Macroaverage'], columns=['Scores'])

    # Print dataframe in tabular format
    print(tabulate(df_score, headers='keys', tablefmt='pretty'))
    print(tabulate(df_average, headers='keys', tablefmt='pretty'))

def update_lexicon(df):
    path = './'
    file_name = 'isk_lexicon.txt'

    lexicon, tuning = open_lexicon(path, file_name)

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

def test_lexicon(df):
    df_truth = df.copy()
    df_predict = df.copy()
    del df_predict['Sentiment']

    df_predict = label(df_predict)

    return accuracy(df_truth, df_predict)

df_train = pd.read_pickle('./isk_train.pkl')
del df_train['id']
df_test = pd.read_pickle('./isk_test.pkl')
del df_test['id']
df_unlabeled = pd.read_pickle('./tuning_isk.pkl')

#cleaned_df = process_dataframe(df_train)
#update_lexicon(cleaned_df)

#test_lexicon(df_test)

cleaned_df = process_dataframe(df_train)
