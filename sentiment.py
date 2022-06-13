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

import matplotlib.pyplot as plt

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

# TODO: report how many changes were made into data

# eng_tester_file = './Data/NLP_English_JAN2022_OPEN.xlsx'
# eng_tester_sheet = 'Result 1'

eng_tester_file = './Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
eng_tester_sheet = 'English'
df = Functions.init(eng_tester_file, eng_tester_sheet)
df = Functions.leave_text(df, "EN")
Functions.label(df)
Accuracy.accuracy('truth.xlsx', 'guess.xlsx')




# for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#     Functions.ADV_VAL = i
#     Functions.label(df)
#     print(str(i) + " : " + str(Accuracy.accuracy('truth.xlsx', 'guess.xlsx')))

# examples = [
#     "one problem - as we were landing there was an announcement that the connecting flight to Toronto was delayed from 5:05 to 6:05 - but it wasn't! That left some of us scrambling when we saw on the board that it was not delayed at all.", 
#     "The plane both ways was very comfortable.  I enjoyed having a pillow.  The staff was truly lovely. I will book Icelandair whenever I can.  Thank you for taking such good care of us",
#     "Kevin at KEF desk was very helpful and helped me to not feel rushed through the process of check in. ",
#     "The fly attendance was Extremely kindly, great job",
#     "For a 6 hour flight I think more drink services should have happened, at least water",
#     "Great pilot's landing in lousy windy weather. Great flight attendants and what a ground crew working in such lousy weather.",
#     "The meal options are minimal, unhealthy and not filling. I don't mind paying, but I'd like a proper meal option.",
#     "All perfect",
#     "The staff on board were excellent. ",
#     "Check in was very good at Manchester but exceptional at KEF.",
#     "no", 
#     "On a long international flight it would be nice if they were snacks served in economy.",
#     "Wifi did not work in flight",
#     "The plane smelled awful"
#     ]
# answer = ['negative', 'positive', 'mix', 'positive', 'negative', 'positive', 'negative', 'positive', 'positive', 'negative', 'neutral', 'negative', 'negative', 'negative']

# eng_dict = Functions.make_dict('eng_lexicon.txt')

# # "Airport process was very efficient just recommend that gates are announced earlier.",
#     # "Airport process was very efficient",
#     # "just recommend that gates are announced earlier.",

# incorrect = [
#     "We loved the flight both ways. Especially enjoyed the lounge."
# ]

# answer = ['positive']

# guess = []
# g = ''

# for text in incorrect:
#     [text, multiplier] = Functions.eng_process(text, 1)
#     indiv_scores = Functions.find_in_dict(text, eng_dict)

#     product = np.multiply(multiplier, indiv_scores)
#     score = sum(product)

#     print(text)
#     print(multiplier)
#     print(indiv_scores)

#     if score > 0:
#         guess.append('positive')
#         g = 'positive'
#     elif score < 0:
#         guess.append('negative')
#         g = 'negative'
#     else:
#         guess.append('neutral')
#         g='neutral'

# print(answer)
# print(guess)