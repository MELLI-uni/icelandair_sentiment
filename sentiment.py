import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain

import re
import numpy as np
import pandas as pd

import Functions
import Accuracy

eng_file_name = './Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet_name = 'Result 1'

# isk_file_name = './Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
# isk_sheet_name = 'Sheet1'

isk_file_name = './Data/NPS_freetext_MAY_2022_Icelandic_answers.xlsx'
isk_sheet_name = 'Icelandic'

#df = Functions.init(isk_file_name, isk_sheet_name)
#df = Functions.clean(df)
#Functions.process(df)


# Accuracy.accuracy(test_actual, test_incorrect)

df = Functions.init(isk_file_name, isk_sheet_name)
Functions.clean_multi(df, "IS")

# df = Functions.init(eng_file_name, eng_sheet_name)
# Functions.clean_multi(df, "EN")

#df = Functions.process(df)
#print(df)

# pattern = '\s\.*,*[Ee]n|[Þþ]ó|[Nn]ema|[Hh]ins vegar\.*,*\s'
# pattern = r'\s*[Tt]akk(\.|fyrir\.)\s*'
# text = "Ófært í sólarhring fyrir flugið og svo bættist við tafirnar vegna bilunar. Takk fyrir. En Upplýsingagjöf var samt með ágætum, það er mikilvægt. !"
# x = re.split(pattern, text)

# print(x)