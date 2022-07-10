import regex as re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from itertools import islice

from reynir import Greynir
from reynir_correct import check

# from translate import Translator

isk_greynir = Greynir()
# translator = Translator(to_lang="is")

# emoji_dict = {}
# with open('../lexicons/emoji.txt', encoding='utf-8') as f:
#     for line in islice(f, 1, None):
#         [key, value] = line.split("\t")
#         emoji_dict[key] = translator.translate(value).strip()
# with open('../lexicons/emoticon.txt', encoding='utf-8') as f:
#     for line in f:
#         [key, value] = line.split("\t")
#         emoji_dict[key] = translator.translate(value).strip()

isk_stop = []
with open('../lexicons/isk_stop.txt', encoding='utf-8') as f:
    for line in f:
        isk_stop.append(line.strip())

# def convert_emoji_emoti(input):
#     """
#     convert_emoji_emoti function replaces all emojis and emoticons in the input with corresponding text descriptions.
#     Emoji descriptions are obtained from: https://emojipedia.org
#     Emoticon descriptions are obtained from: https://en.wikipedia.org/wiki/List_of_emoticons

#     The lexicon for emoji and emoticon are located under '../lexicons/' folder under the name emoji.txt and emoticon.txt correspondingly
#     The emoji and emoticon lexicon can be updated by calling the 'update_emoji' or 'update_emoticon' function in the rulebase_test.py file

#     : param input: input sentence that will have the emoji converted

#     : return: sentence with emoji and emoticons converted into text
#     """

#     converted_input = input

#     tokens = converted_input.split(" ")
#     for token in tokens:
#         if len(token) > 4 or len(token) == 1:
#             continue
#         elif token.lower() in STOPWORDS:
#             continue
#         elif token in emoji_dict:
#             converted_input = converted_input.replace(token, (" " + emoji_dict[token] + " "))

#     converted_input = converted_input.replace("  ", " ")

#     for item in (re.findall(r'[^\w\s]', input)):
#         if item in emoji_dict:
#             converted_input = converted_input.replace(item, (" " + emoji_dict[item] + " "))
#             continue
#         converted_input = converted_input.replace(item, "")

#     converted_input = converted_input.replace("  ", " ")

#     return converted_input

df_isk = pd.read_pickle('./isk_data.pkl')

# df_isk = df_isk['answer_freetext_value'].apply(lambda x: convert_emoji_emoti(x), axis=1)