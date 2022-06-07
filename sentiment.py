import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import chain

import re
import numpy as np
import pandas as pd
import spacy

import Update
import Functions
import Accuracy

eng_file_name = './Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet_name = 'Result 1'

# isk_file_name = './Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
# isk_sheet_name = 'Sheet1'

isk_file_name = './Data/NPS_freetext_MAY_2022_Icelandic_answers.xlsx'
isk_sheet_name = 'Icelandic'

flight_lexicon = Functions.load_lexicon()

df = Functions.init(isk_file_name, isk_sheet_name)
df = Functions.clean_multi(df, "EN")

# df = Functions.init(eng_file_name, eng_sheet_name)
# df = Functions.clean_multi(df, "EN")
# df = Functions.process(df, "EN", flight_lexicon)
# Update.eng_update(df)

# sample_sentence = "Every was wonderful.  My husband said he'll only go places that Iceland Air flies."
# dictionary = Update.eng_dict()
# cleaned = Functions.process_sentence(sample_sentence, "EN", flight_lexicon)
# score = Functions.calculate(cleaned, "EN", dictionary)

# if score >= 0.1:
#     print('positive')
# elif score <= -0.1:
#     print('negative')
# else:
#     print('neutral')