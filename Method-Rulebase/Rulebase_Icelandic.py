import regex as re
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from itertools import islice

from reynir import Greynir
from reynir_correct import check

isk_greynir = Greynir()

isk_stop = []
with open('../../lexicons/isk_stop.txt', encoding='utf-8') as f:
    for line in f:
        isk_stop.append(line.strip())

df_train = pd.read_pickle('./isk_train.pkl')
df_test = pd.read_pickle('./isk_test.pkl')
df_unlabled = pd.read_pickle('./tuning_isk.pkl')