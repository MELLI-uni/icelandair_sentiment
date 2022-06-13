import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain

import regex as re
import numpy as np
import pandas as pd
import spacy

import UpdateLanguage
import Functions
import Accuracy

# isk_file_name = './Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
# isk_sheet_name = 'Sheet1'
# isk_file_name = './Data/NPS_freetext_MAY_2022_Icelandic_answers.xlsx'
# isk_sheet_name = 'Icelandic'

eng_file_name = './Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet_name = 'Result 1'
df = Functions.init(eng_file_name, eng_sheet_name)
df = Functions.clean(df, "EN")
df = Functions.process(df, "EN")

# #df = df.sample(frac=1).reset_index(drop=True)   # Check how much more 'correctly' labeled data needed
UpdateLanguage.update_lexicons(df)

# df_temp = df[:400]
# UpdateLanguage.update_lexicons(df_temp)

# df_temp = df[:800]
# UpdateLanguage.update_lexicons(df_temp)

# df_temp = df[801:]
# UpdateLanguage.update_lexicons(df_temp)

eng_file_name = './Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
eng_sheet_name = 'English'
df = Functions.init(eng_file_name, eng_sheet_name)
df = Functions.clean(df, "EN")
df = Functions.process(df, "EN")
df = UpdateLanguage.update_lexicons(df)

# df = Functions.init(isk_file_name, isk_sheet_name)
# df = Functions.clean(df, "IS")
# df = Functions.process(df, "IS")
# df = UpdateLanguage.update_lexicons(df)

# eng_dict = Functions.make_dict('eng_lexicon.txt')

# eng_tester_file = './Data/NLP_English_JAN2022_OPEN.xlsx'
# eng_tester_sheet = 'Result 1'

eng_tester_file = './Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
eng_tester_sheet = 'English'
df = Functions.init(eng_tester_file, eng_tester_sheet)
df = Functions.leave_text(df, "EN")
Functions.label(df)
Accuracy.accuracy('truth.xlsx', 'guess.xlsx')
