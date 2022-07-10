import regex as re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from reynir import Greynir
from reynir_correct import check

from translate import Translator

isk_greynir = Greynir()
translator = Translator(to_lang="is")

emoji_dict = {}


translation = translator.translate("this is a pen")
print(translation)
