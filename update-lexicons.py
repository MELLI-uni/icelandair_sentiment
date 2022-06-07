import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain

import re
import numpy as np
import pandas as pd

import Functions

from gensim.parsing.preprocessing import STOPWORDS
STOPWORDS = STOPWORDS.union(set(['icelandair']))

# TASK2-D: Create an update-lexicon py file that will create eng-lexicons based on the leftover words
def eng_update(df):
    eng_lexicon = "./lexicons/eng_lexicon.txt"

    eng_dict = {}

    f = open(eng_lexicon, 'r+', encoding='utf-8')

    for lines in f:
        [key, skip, scores] = lines.split('\t')
        eng_dict[key] = [int(x) for x in scores.strip('\n[]').split(', ')]

    f.seek(0)

    for row in df.itertuples(index=False):
        new_word = row.Change
        new_score = row.Sentiment

        if new_word in eng_dict:
            eng_dict[new_word].append(new_score)
        else:
            eng_dict[new_word] = [new_score]

    for key, value in eng_dict.items():
        f.write(key + "\t" + str(sum(value) / len(value)) + "\t" + str(value) + "\n")

    f.close()

# TASK2-E: Create an update-lexicon py file that will create eng-lexicons based on the leftover words
def isk_update():
    print("Icelandic update")

sample_location = './Tester_Files/sample-processed.xlsx'
sample_df = pd.read_excel(sample_location, header=0)
del sample_df['Unnamed: 0']
del sample_df['id']
#print(sample_df)

eng_update(sample_df)