# import nltk
# from nltk import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from itertools import chain

# import regex as re
# import numpy as np
import pandas as pd
# import spacy

import UpdateLanguage
import Functions
# import Accuracy

# # isk_file_name = './Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
# # isk_sheet_name = 'Sheet1'
# # isk_file_name = './Data/NPS_freetext_MAY_2022_Icelandic_answers.xlsx'
# # isk_sheet_name = 'Icelandic'

eng_file_name = '../Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet_name = 'Result 1'
df = Functions.init(eng_file_name, eng_sheet_name)
df.to_excel("temp.xlsx")
#df = Functions.clean(df, "EN")
#df = Functions.process(df, "EN")

# # #df = df.sample(frac=1).reset_index(drop=True)   # Check how much more 'correctly' labeled data needed
# UpdateLanguage.update_lexicons(df)

# # df_temp = df[:400]
# # UpdateLanguage.update_lexicons(df_temp)

# # df_temp = df[:800]
# # UpdateLanguage.update_lexicons(df_temp)

# # df_temp = df[801:]
# # UpdateLanguage.update_lexicons(df_temp)

# eng_file_name = './Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
# eng_sheet_name = 'English'
# df = Functions.init(eng_file_name, eng_sheet_name)
# df = Functions.clean(df, "EN")
# df = Functions.process(df, "EN")
#df = UpdateLanguage.update_lexicons(df)

# # df = Functions.init(isk_file_name, isk_sheet_name)
# # df = Functions.clean(df, "IS")
# # df = Functions.process(df, "IS")
# # df = UpdateLanguage.update_lexicons(df)

# # eng_dict = Functions.make_dict('eng_lexicon.txt')

# # eng_tester_file = './Data/NLP_English_JAN2022_OPEN.xlsx'
# # eng_tester_sheet = 'Result 1'

# eng_tester_file = './Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
# eng_tester_sheet = 'English'
# df = Functions.init(eng_tester_file, eng_tester_sheet)
# df = Functions.leave_text(df, "EN")
# Functions.label(df)
# Accuracy.accuracy('truth.xlsx', 'guess.xlsx')

# import spacy
# from spacy.lang.en import English
# import regex as re

# nlp = spacy.load("en_core_web_sm")
# nlp2 = English()
# tokenizer = nlp2.tokenizer
# doc = nlp("I didn't buy food on board but my friend traveling with me would have done if the choice had been better.")
# doc = nlp("All very efficient, pleasant and attentive - friendly! ")
# doc = nlp("It would've been nice to know that we could not check in earlier")

# degree_adv = []


# TODO: extract all (ADV advmod) from the sentence and store it on a separate lexicon degree adv list
# if in format AUX AUX ADJ make the ADJ to opposite score
# if ADJ CCONJ ADJ format with advmod before, apply advmod intensifier to both adjs
# then eliminate cconj

# order of importance: adj -> adv -> verb -> noun

# Reference: 
# https://en.wiktionary.org/wiki/Category:Icelandic_prefixes
# https://en.wiktionary.org/wiki/Category:English_productive_prefixes
# https://en.wiktionary.org/wiki/Category:English_unproductive_prefixes
# https://en.wiktionary.org/wiki/Category:English_productive_suffixes
# https://en.wiktionary.org/wiki/Category:English_unproductive_suffixes

# TODO: for english eliminate prefix and suffix -> make defintion
# TODO: for icelandic eliminate prefix 
# TODO: for both language search for match in emoji or emoticon and eliminate

#text = "I didn't buy food on board but my friend traveling with me would have done if the choice had been better."
#text = "Very uncomfortable seat cushion"
#text = "I did not like my seat"
#text = "Re-evaluate the training of your hostesses."
# text = "We'll be back! Much appreciated!"
#text = "Would like a USB-C connection to the screens/entertainment. Many headphones use these connections now."
#text = "It would be better if your baggage allowance was clear on the ticket rather than having to go back in to check on the web via the ticket type.  "
#text = "It was bad because your baggage allowance was not clear on the ticket"
#text = "It would have been good if only I can get a meal on the plane"
#text = "The plane was on time"
#text = "I hated the flight because the plane was so hot"
#text = "I would love to fly with you again"
#text = "training is needed for the staff"
#text = "This was one of the nicest, efficient, and pleasant flight crew we ever have experienced."

# [tokens, score] = clean(text, -1)
# print(degree_adv)
# print(tokens)
# print(score)