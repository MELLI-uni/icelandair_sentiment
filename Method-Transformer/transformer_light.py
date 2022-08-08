import pandas as pd
import Light_Transformer

df_eng = pd.read_pickle('../Data/eng_total.pkl')
del df_eng['id']

df_isk = pd.read_pickle('../Data/isk_total.pkl')
del df_isk['id']

