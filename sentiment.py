import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

import Functions
import Accuracy

eng_file_name = './Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet_name = 'Result 1'

isk_file_name = './NLP_Icelandic_14FEB2022_YTD.xlsx'
isk_sheet_name = 'Result 1'

#df = Functions.init(eng_file_name, eng_sheet_name)

#df = Functions.clean(df)

#Functions.process(df)

# test_actual = "./Tester_Files/dummy_actual.xlsx"
# test_correct = "./Tester_Files/dummy_correct.xlsx"
# test_incorrect = "./Tester_Files/dummy_incorrect.xlsx"

# 
# Accuracy.accuracy(test_actual, test_incorrect)

#print(df)