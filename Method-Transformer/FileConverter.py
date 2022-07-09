import pandas as pd
import numpy as np
import regex as re

import nltk
from nltk.tokenize import sent_tokenize

from reynir import Greynir

def DFtoTXT(df, file_name):
    folder_path = '/home/jiyoon/work/Data/Tuning/'
    file_path = folder_path + file_name

    np.savetxt(file_path, df.values, fmt='%s', newline='\n', encoding='utf-8')

def SplitBySentenceEnglish(file_name):
    sentences = []

    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    f.close()

    for line in lines:
        if re.search('\w', line) != None:
            sentences.extend(sent_tokenize(line.rstrip()))

    with open('tuning_eng_tokenized.txt', 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s + "\n")
    f.close()

def SplitBySentenceIcelandic(file_name):
    sentences = []
    
    g = Greynir()

    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    f.close()

    for line in lines:
        j = g.submit(line)
        for pg in j.paragraphs():
            for sent in pg:
                sentences.append(sent.tidy_text)

    with open('tuning_isk_tokenized.txt', 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s + "\n")
    f.close()

#text_file = 'tuning_eng.txt'
#SplitBySentence(text_file)

text_file = 'tuning_isk.txt'
SplitBySentenceIcelandic(text_file)
