import regex as re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from itertools import islice

from reynir import Greynir
from reynir_correct import check

from translate import Translator

isk_greynir = Greynir()
translator = Translator(to_lang="is")

emoji_dict = {}
with open('../lexicons/emoji.txt', encoding='utf-8') as f:
    for line in islice(f, 1, None):
        [key, value] = line.split("\t")
        emoji_dict[key] = translator.translate(value).strip()
with open('../lexicons/emoticon.txt', encoding='utf-8') as f:
    for line in f:
        [key, value] = line.split("\t")
        emoji_dict[key] = translator.translate(value).strip()