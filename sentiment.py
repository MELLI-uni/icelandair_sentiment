import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

import Functions

eng_file_name = './Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet_name = 'Result 1'

df = Functions.init(eng_file_name, eng_sheet_name)

df = Functions.clean(df)

Functions.process(df)
print(df)