import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain

import numpy as np
import pandas as pd

import Functions

eng_file_name = './Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet_name = 'Result 1'

df = Functions.init(eng_file_name, eng_sheet_name)

Functions.clean_multi(df, "EN")

#df = Functions.process(df)
#print(df)