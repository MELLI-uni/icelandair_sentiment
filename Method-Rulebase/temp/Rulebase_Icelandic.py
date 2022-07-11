import regex as re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from itertools import islice

from reynir import Greynir
from reynir_correct import check_single

g = Greynir()

isk_stop = []
#with open('../../lexicons/isk_stop.txt', encoding='utf-8') as f:
#    for line in f:
#        isk_stop.append(line.strip())

def data_cleaning(df):
    # Remove na values from dataframe
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Remove duplicate values from dataframe
    df = df.drop_duplicates()

    sentences = []

    for line in df.values:
        j = g.submit(line[0])
        for pg in j.paragraphs():
            for sent in pg:
                sentences.append(sent.tidy_text)
                sent.parse()

    return sentences

df_train = pd.read_pickle('./isk_train.pkl')
df_test = pd.read_pickle('./isk_test.pkl')
df_unlabeled = pd.read_pickle('./tuning_isk.pkl')

sentences_list = data_cleaning(df_unlabeled)
print(sentences_list[0])
